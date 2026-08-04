#!/usr/bin/env bash
# Run the subset of pre-commit hooks that are pure-Python and pass on macOS.
# Skips hooks needing external binaries (clang-format, shellcheck, cargo, ...)
# and the pip-compile hooks, which want a CUDA-ish environment.
set -euo pipefail

HOOKS=(
  ruff-check
  ruff-format
  typos
  check-spdx-header
  check-filenames
  check-root-lazy-imports
)

echo "--- Creating venv"
uv venv --clear --python 3.12 .venv-bk
# shellcheck disable=SC1091
source .venv-bk/bin/activate
uv pip install -r requirements/lint.txt

echo "--- Running hooks against origin/main..HEAD"
git fetch -q origin main
for hook in "${HOOKS[@]}"; do
  echo "+++ pre-commit: ${hook}"
  pre-commit run "${hook}" --from-ref origin/main --to-ref HEAD --show-diff-on-failure
done
