#!/usr/bin/env bash
# Build the persistent test-dependency venv on the GPU test host.
#
# The vllm/vllm-openai images are slim serving images: they ship pytest but none
# of the plugins upstream's commands rely on, and none of the libraries the tests
# import. Rather than build a custom image (the docker filesystem on that host
# has no room), the extras live in a venv on the host that every test container
# mounts at /test-venv and picks up through PYTHONPATH.
#
# Run this inside the test image, not on the host: the venv is created with the
# in-container path so its scripts' shebangs point at /test-venv/bin/python.
# Running it from the host produces a venv whose `pip` dies with
# "bad interpreter: /test-venv/bin/python3: No such file or directory".
#
#   docker run --rm --entrypoint bash \
#     -v /models/chengfeng-ci/test-venv:/test-venv \
#     -v "$PWD/.buildkite:/bk:ro" \
#     vllm/vllm-openai:nightly /bk/local-scripts/setup-test-venv.sh
#
# The host path is on /models because the root filesystem is at 95%. Only the
# in-container path matters to the venv itself: pyvenv.cfg and every shebang
# record /test-venv, so relocating the host side is a plain mv.
#
# Idempotent: re-running upgrades in place. Safe to re-run after the image
# changes, which is the point - without this script the venv's contents are
# whatever someone installed by hand, and a rebuild silently loses them.
#
# This is NOT part of vLLM upstream CI.
set -euo pipefail

VENV="${TEST_VENV:-/test-venv}"
# The host cannot reach pypi.org reliably; see the HF_ENDPOINT note in
# ci_config_4090.yaml for the same problem with huggingface.co.
INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "--- Creating venv at $VENV"
  # --system-site-packages so torch and the compiled vllm still come from the
  # image. The image's torch already matches what requirements/test/cuda.in
  # pins, so nothing below overwrites it.
  python3 -m venv --system-site-packages "$VENV"
fi

echo "--- Installing test dependencies"
# Two groups, both discovered by running the generated pipeline and reading what
# the failures asked for:
#
#   pytest plugins   upstream's commands pass --shard-id, --timeout and --forked,
#                    and tblib is needed to pickle exceptions across processes.
#   test imports     libraries the test modules import directly. ray also backs
#                    the distributed executor tests; multiprocess is only pinned
#                    in requirements/test/rocm.in but tests/distributed imports
#                    it on every platform. torch-abi-audit is imported by
#                    .buildkite/check-torch-abi.py, not by a test module, which is
#                    why it surfaced late: fail_fast kept cancelling that step
#                    before it ever ran. pqdm, av and soundfile surfaced the same
#                    way in build #29 - the steps needing them had never reached
#                    collection, so "run the tests and see what is missing" could
#                    not have found them earlier. All three are pinned upstream in
#                    requirements/test/cuda.in.
"$VENV/bin/python" -m pip install --quiet --no-cache-dir --index-url "$INDEX" \
  pytest-asyncio \
  pytest-shard \
  pytest-timeout \
  pytest-forked \
  pytest-rerunfailures \
  tblib \
  'ray[cgraph,default]>=2.48.0' \
  'multiprocess==0.70.16' \
  'lm-eval[api]>=0.4.12' \
  'torch-abi-audit==0.0.1' \
  'pqdm==0.2.0' \
  'av==16.1.0' \
  'soundfile==0.12.1'

# ray[cgraph] pulls cupy-cuda12x, but the image is CUDA 13 and ships
# cupy-cuda13x. Both end up importable and the venv's copy wins through
# PYTHONPATH, so cupy loads a CUDA 12 build against a CUDA 13 runtime. cupy
# itself warns about it ("multiple CuPy packages are installed") and keeps
# going. Drop the wrong one and let the image's own build show through.
if "$VENV/bin/python" -m pip show cupy-cuda12x >/dev/null 2>&1; then
  echo "--- Removing cupy-cuda12x (conflicts with the image's cupy-cuda13x)"
  "$VENV/bin/python" -m pip uninstall --quiet --yes cupy-cuda12x
fi

echo "--- Verifying"
"$VENV/bin/python" - <<'PY'
import importlib

for mod in ("pytest_asyncio", "pytest_shard", "pytest_timeout", "pytest_forked",
            "tblib", "ray", "multiprocess", "lm_eval", "torch_abi_audit",
            "pqdm", "av", "soundfile"):
    try:
        importlib.import_module(mod)
        print(f"  ok    {mod}")
    except ImportError as exc:
        raise SystemExit(f"  MISSING {mod}: {exc}")
PY

echo "--- Done. Size: $(du -sh "$VENV" | cut -f1)"
