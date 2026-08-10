#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Write the files that only an offline HF cache needs.

`AutoTokenizer.from_pretrained` reads `config.json` even for a repo that is
purely a tokenizer. Online that is harmless: transformers special-cases a
missing `config.json` (see `filenames != "config.json"` in its `cached_files`)
and carries on. Offline the same absence surfaces as LocalEntryNotFoundError
wrapped in OSError, because huggingface_hub cannot tell "the Hub returned 404"
from "this file is not in the cache" - so a repo that legitimately has no
config becomes indistinguishable from one we failed to download.

The fix is to give the offline path something to read. An empty object is
enough: everything AutoTokenizer needs is already in `tokenizer_config.json`
(`tokenizer_class`), and an empty config contributes no fields, so the
resulting tokenizer is the one the online path would have built.

This is not a substitute for downloading a model. It applies only where the
upstream repo has no `config.json` at all, verified with `list_repo_files`.
If a repo does publish one, the absence is a download gap and belongs there.

Not part of vLLM upstream CI.

Usage (on the test host, after the models are staged):

    HF_HOME=/models/chengfeng-test /usr/bin/python3 stage_offline_shims.py
"""

import os
import pathlib
import sys

# Repos that publish no config.json upstream, and the reason each is reached.
# Confirmed per repo with HfApi().list_repo_files - do not add an entry without
# checking, or a genuinely missing download gets papered over.
NO_CONFIG_REPOS = {
    # Reached as the tokenizer= of moondream/moondream3-preview in
    # tests/models/registry.py. Holds 3 tokenizer files and nothing else.
    "moondream/starmie-v1": "tokenizer of moondream/moondream3-preview",
}

HF_HUB = pathlib.Path(os.environ.get("HF_HOME", "/models/chengfeng-test")) / "hub"


def stage(repo_id: str) -> str:
    """Write an empty config.json into `repo_id`'s cached snapshot."""
    root = HF_HUB / ("models--" + repo_id.replace("/", "--"))
    snapshots = sorted((root / "snapshots").glob("*")) if root.exists() else []
    if not snapshots:
        return "SKIP  not cached - download the repo first"

    # A repo can have several snapshots cached; the tests resolve through
    # refs/main, but writing to each is harmless and keeps this independent of
    # which revision a caller asks for.
    for snapshot in snapshots:
        target = snapshot / "config.json"
        if target.exists():
            return "SKIP  already has config.json"
        target.write_text("{}\n")
    return "OK    wrote %d config.json" % len(snapshots)


def main() -> int:
    if not HF_HUB.is_dir():
        print(f"{HF_HUB} does not exist - is HF_HOME right?", file=sys.stderr)
        return 1

    failed = False
    for repo_id, why in NO_CONFIG_REPOS.items():
        try:
            status = stage(repo_id)
        except OSError as exc:
            status = f"FAIL  {type(exc).__name__}: {exc}"
            failed = True
        print(f"{status}  {repo_id}  ({why})")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
