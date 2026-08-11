#!/usr/bin/env bash
# Run the handful of unit tests that work without a compiled vLLM.
#
# tests/conftest.py imports torch and vllm, so it cannot be loaded on a machine
# without a built vLLM extension. --noconftest skips it, which limits us to
# tests that need no fixtures from it. Each file below was verified to pass on
# macOS/arm64 with nothing but pytest and cmake installed.
set -euo pipefail

TESTS=(
  tests/test_cmake_utils.py
  tests/tools/test_config_validator.py
  tests/tools/test_docker_build_metadata_args.py
  tests/v1/executor/test_multiproc_executor_timeout.py
  tests/v1/kv_connector/unit/test_moriio_proxy_routing.py
)

# Keep the venv outside the repo: the agent runs `git clean -ffxdq` before every
# job, which would otherwise delete it and force a reinstall each time.
VENV="$HOME/.cache/vllm-bk-unit-venv"
echo "--- Preparing venv at ${VENV}"
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -q pytest cmake pyyaml regex

echo "--- Running unit tests"
python -m pytest "${TESTS[@]}" \
  --noconftest \
  -p no:cacheprovider \
  -v -ra

# The generator's own pure-function checks live as doctests, so they run here
# rather than needing a test file of their own. Currently the CUDA-arch coverage
# rule, which decides whether an image can run on this box at all.
echo "--- Generator doctests"
python -m doctest .buildkite/local-scripts/gen_pipeline.py && echo "doctests passed"
