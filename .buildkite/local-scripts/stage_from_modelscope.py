#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage HF-gated models from ModelScope into the HF cache layout.

Two models the tests need are gated on huggingface.co and this account has no
grant for them, so every file request returns 403 (401 without a token):

    meta-llama/Llama-3.2-1B    tests/v1/engine/utils.py:24 TOKENIZER_NAME
    google/gemma-3-1b-it       several model tests

ModelScope mirrors both without gating. The catch is layout: ModelScope writes
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
}

HF_HUB = pathlib.Path(os.environ.get("HF_HOME", "/models/chengfeng-test")) / "hub"
STAGING = pathlib.Path("/models/chengfeng-ci/ms-staging")

# See the module docstring: any fixed string works offline as long as refs/main
# names it.
REVISION = "modelscope-mirror"

# ModelScope's own bookkeeping. The HF loader neither needs nor expects these.
MS_METADATA = {"._____temp", ".mdl", ".msc", ".mv"}


def download(ms_id: str) -> pathlib.Path:
    from modelscope import snapshot_download

    return pathlib.Path(snapshot_download(ms_id, cache_dir=str(STAGING)))


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
            src = download(ms_id)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAIL {hf_id}  download: {type(exc).__name__}: {exc}")
            failed.append(hf_id)
            continue
        count = place(hf_id, src)
        print(f"OK   {hf_id}  {count} files  (from {ms_id})")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
