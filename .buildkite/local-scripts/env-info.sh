#!/usr/bin/env bash
# Sanity check that the local agent has the toolchain the other steps need.
set -euo pipefail

echo "--- Agent context"
echo "host:     $(hostname)"
echo "arch:     $(uname -m)"
echo "cwd:      $(pwd)"
echo "branch:   ${BUILDKITE_BRANCH:-<none>}"
echo "commit:   ${BUILDKITE_COMMIT:-<none>}"
echo "pr:       ${BUILDKITE_PULL_REQUEST:-false}"

echo "--- Toolchain"
command -v uv >/dev/null || { echo "uv not found on PATH"; exit 1; }
uv --version
git --version
