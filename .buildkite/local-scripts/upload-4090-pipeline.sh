#!/usr/bin/env bash
# Generate a test pipeline and append it to the current Buildkite build.
#
# Runs on the macOS bootstrap agent. The steps it uploads target the queue named
# in the config, so Buildkite dispatches them to that GPU agent - the two agents
# share nothing but the build.
#
# Which platform is selected for is entirely the config's business. Override it
# to target a different agent:
#   CI_CONFIG=.buildkite/ci_config_h200.yaml .buildkite/local-scripts/upload-4090-pipeline.sh
#
# This is NOT part of vLLM upstream CI.
set -euo pipefail

# CI_CONFIG_4090 is the older name, still honoured so existing pipeline steps
# that set it keep working.
CONFIG="${CI_CONFIG:-${CI_CONFIG_4090:-.buildkite/ci_config_4090.yaml}}"
GENERATOR=".buildkite/local-scripts/gen_pipeline.py"

# The generator diffs against the merge base, which needs the base branch to
# exist locally. Buildkite clones shallow, so fetch it first.
BASE_BRANCH="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"
if [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]]; then
  BASE_BRANCH="main"
fi

git config --global --add safe.directory "$(pwd)" 2>/dev/null || true

echo "--- Fetching origin/${BASE_BRANCH} for the diff base"
git fetch --no-tags --depth=50 origin \
  "${BASE_BRANCH}:refs/remotes/origin/${BASE_BRANCH}" 2>/dev/null ||
  git fetch --no-tags origin \
    "${BASE_BRANCH}:refs/remotes/origin/${BASE_BRANCH}" 2>/dev/null ||
  echo "WARNING: could not fetch origin/${BASE_BRANCH}; generator will run-all"

# PyYAML and regex are the generator's only dependencies; regex is required
# because check_forbidden_imports.py forbids the stdlib `re`. Keep the venv
# outside the repo so the agent's `git clean -ffxdq` does not delete it
# between jobs.
VENV="$HOME/.cache/vllm-bk-gen-venv"
if [[ ! -d "$VENV" ]]; then
  echo "--- Creating generator venv at ${VENV}"
  uv venv --python 3.12 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -q pyyaml regex

# Name outputs after the config, so several platform configs can run in the same
# build without overwriting each other's artifacts.
PLATFORM="$(basename "$CONFIG" .yaml | sed 's/^ci_config_//')"
OUT="/tmp/pipeline-${PLATFORM}.yaml"
REPORT="/tmp/selection-${PLATFORM}.txt"

# One generator run produces both: the pipeline on stdout, the report to
# --report. The report is this agent's output, not the test agent's - nothing in
# the uploaded pipeline depends on it - so it travels as a build artifact.
echo "--- Generating pipeline"
python3 "$GENERATOR" --config "$CONFIG" --report "$REPORT" >"$OUT"
if [[ ! -s "$OUT" ]]; then
  echo "generator produced no pipeline"
  exit 1
fi

# `+++` marks the section Buildkite expands by default, so the selection is the
# first thing visible on the step. The headline goes in the header itself, so the
# verdict reads without expanding anything. Read from the report rather than
# re-running the generator: classification walks hundreds of test files.
HEADLINE="$(head -1 "$REPORT")"
echo "+++ Selection: ${HEADLINE:-see below}"
cat "$REPORT"

# Upload both before the pipeline upload, so the report survives a rejected
# pipeline. TODO: also ship $REPORT to S3.
if command -v buildkite-agent >/dev/null 2>&1; then
  buildkite-agent artifact upload "$REPORT" || true
  buildkite-agent artifact upload "$OUT" || true
  echo "--- Uploading pipeline"
  # --no-interpolation: the pipeline references $BUILDKITE_BUILD_CHECKOUT_PATH,
  # which must resolve on the *test* agent. Without this the upload interpolates
  # it here, baking in this machine's checkout path (a macOS path under
  # /opt/homebrew) that does not exist on the GPU host, and every mount fails.
  buildkite-agent pipeline upload --no-interpolation "$OUT"
else
  echo "buildkite-agent not on PATH; generated pipeline follows"
  cat "$OUT"
fi
