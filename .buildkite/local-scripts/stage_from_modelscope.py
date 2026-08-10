#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage HF-gated models from ModelScope into the HF cache layout.

Some models the tests need are gated on huggingface.co without a grant for this
account, so every file request returns 403 (401 when no token is sent at all).
That distinction is the useful one: 401 means fix the token, 403 means the token
is fine and someone has to accept a licence on the Hub. Only the 403 set belongs
here.

ModelScope mirrors them without gating. The catch is layout: ModelScope writes
`<cache>/<org>/<name>/`, while the tests ask huggingface_hub for a repo id and
it looks under `$HF_HOME/hub/models--<org>--<name>/snapshots/<rev>/`, with the
revision read from `refs/main`. So download, then place.

Placement uses hardlinks, not copies: staging dir and HF cache are both on
/models, and a 4.7G model does not need a second copy. Deleting either side
leaves the other intact.

The revision is a fixed sentinel rather than the real commit sha. Offline,
huggingface_hub only needs refs/main and snapshots/<that> to agree - it does
not check the value against the Hub. A test passing an explicit `revision=`
would miss this, but none of the ones we run do.

Not part of vLLM upstream CI.

Usage (on the test host, needs MODELSCOPE_API_TOKEN for the download half):

    MODELSCOPE_API_TOKEN=ms-... /usr/bin/python3 stage_from_modelscope.py
"""

import os
import pathlib
import sys

# HF repo id -> ModelScope repo id. The HF id is what the tests ask for and so
# what the cache directory must be named after; the ModelScope id is only how
# we obtain the bytes.
MODELS = {
    "meta-llama/Llama-3.2-1B": "LLM-Research/Llama-3.2-1B",
    "google/gemma-3-1b-it": "LLM-Research/gemma-3-1b-it",
    # A second batch, same reason. These came back 403 rather than 401 with a
    # valid token, meaning the token works but this account holds no grant for
    # the repo - so waiting on a licence click is the only fix on the HF side.
    # ModelScope mirrors them ungated. Note the id is not always under
    # LLM-Research: two keep the original org name.
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": (
        "LLM-Research/Llama-4-Scout-17B-16E-Instruct"
    ),
    "meta-llama/Llama-Guard-4-12B": "LLM-Research/Llama-Guard-4-12B",
    "meta-llama/Meta-Llama-3-8B-Instruct": "LLM-Research/Meta-Llama-3-8B-Instruct",
    "facebook/chameleon-7b": "facebook/chameleon-7b",
    "CohereLabs/command-a-vision-07-2025": "CohereLabs/command-a-vision-07-2025",
    # nvidia/Eagle2.5-8B has no ModelScope mirror under any org tried, so it
    # stays uncovered until the HF grant comes through.
}

HF_HUB = pathlib.Path(os.environ.get("HF_HOME", "/models/chengfeng-test")) / "hub"
STAGING = pathlib.Path("/models/chengfeng-ci/ms-staging")

# See the module docstring: any fixed string works offline as long as refs/main
# names it.
REVISION = "modelscope-mirror"

# ModelScope's own bookkeeping. The HF loader neither needs nor expects these.
MS_METADATA = {"._____temp", ".mdl", ".msc", ".mv"}


# Models whose tests only build a config or tokenizer, never load weights. For
# these the weights are pure cost - Llama-4-Scout alone is several hundred GB -
# so fetch metadata only. Everything absent from this set is downloaded whole,
# which is what the models that do run inference need.
CONFIG_ONLY = {
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-Guard-4-12B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "facebook/chameleon-7b",
    "CohereLabs/command-a-vision-07-2025",
}

CONFIG_PATTERNS = ["*.json", "*.txt", "*.model", "*.py"]


def download(ms_id: str, config_only: bool = False) -> pathlib.Path:
    from modelscope import snapshot_download

    kwargs = {"allow_patterns": CONFIG_PATTERNS} if config_only else {}
    return pathlib.Path(snapshot_download(ms_id, cache_dir=str(STAGING), **kwargs))


def place(hf_id: str, src: pathlib.Path) -> int:
    """Hardlink `src`'s files into the HF cache layout for `hf_id`."""
    root = HF_HUB / ("models--" + hf_id.replace("/", "--"))
    snapshot = root / "snapshots" / REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    (root / "refs").mkdir(parents=True, exist_ok=True)
    (root / "refs" / "main").write_text(REVISION)

    linked = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if rel.parts[0] in MS_METADATA:
            continue
        dst = snapshot / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.link(path, dst)
        linked += 1
    return linked


def main() -> int:
    if not os.environ.get("MODELSCOPE_API_TOKEN"):
        print("MODELSCOPE_API_TOKEN is not set", file=sys.stderr)
        return 1

    failed = []
    for hf_id, ms_id in MODELS.items():
        try:
            src = download(ms_id, config_only=hf_id in CONFIG_ONLY)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAIL {hf_id}  download: {type(exc).__name__}: {exc}")
            failed.append(hf_id)
            continue
        count = place(hf_id, src)
        print(f"OK   {hf_id}  {count} files  (from {ms_id})")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
