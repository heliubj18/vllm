#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Find and cache model repositories referenced by vLLM CI tests.

Examples:
  python tools/download_ci_models.py --list
  python tools/download_ci_models.py --changed --download --source huggingface
  python tools/download_ci_models.py --model Qwen/Qwen2.5-1.5B-Instruct --download

Downloads are resumable through huggingface_hub/modelscope. A successful
download writes a small marker in the destination directory; incomplete
directories are passed back to the downloader so interrupted files can resume.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import regex as re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = Path(os.environ.get("VLLM_CI_MODEL_DIR", ROOT / ".ci-models"))
MARKER = ".download-complete"

TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".txt",
    ".md",
    ".toml",
    ".in",
}
SKIP_PARTS = {".git", ".ci-models", "output", "__pycache__", ".venv"}

# These patterns intentionally cover the calling conventions used by tests,
# CI shell scripts, and lm-eval configuration files. The final validator keeps
# ordinary URLs, local paths, and Python symbols out of the result.
MODEL_PATTERNS = (
    re.compile(
        r"(?:from_pretrained|snapshot_download|hf_api\(\)\.snapshot_download)\(\s*[\"']([^\"']+)[\"']"
    ),
    re.compile(
        r"(?:model|model_id|model_name|MODEL_NAME|MODEL_PATH)\s*[:=]\s*[\"']([^\"']+)[\"']"
    ),
    re.compile(
        r"(?:--model|-m)\s+([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)"
    ),
    re.compile(
        r"(?:^|[\s,])model:\s*([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)",
        re.MULTILINE,
    ),
)


@dataclass
class ModelRef:
    model_id: str
    files: set[str] = field(default_factory=set)


def git_changed_files() -> list[Path]:
    """Return files changed against the PR base or origin/main."""
    base = os.environ.get("BUILDKITE_PULL_REQUEST_BASE_BRANCH", "main")
    if os.environ.get("BUILDKITE_PULL_REQUEST", "false") == "false":
        base = "main"
    result = subprocess.run(
        ["git", "diff", "--name-only", f"origin/{base}...HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cannot find diff against origin/{base}; fetch the base branch first"
        )
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def candidate_files(changed: bool, scan_all: bool) -> list[Path]:
    if changed:
        paths = git_changed_files()
    else:
        paths = [p for p in ROOT.rglob("*") if p.is_file()]
    paths = sorted(
        p
        for p in paths
        if p.suffix.lower() in TEXT_SUFFIXES
        and not any(part in SKIP_PARTS for part in p.relative_to(ROOT).parts)
    )
    if scan_all:
        return paths
    # The CI case inventory is under tests/ and .buildkite/. Scanning docs and
    # examples by default would turn every documentation example into a model
    # download request, even when no CI step references it.
    return [p for p in paths if p.relative_to(ROOT).parts[0] in {"tests", ".buildkite"}]


def valid_model_id(value: str) -> bool:
    value = value.strip().strip("`$()")
    if value.startswith(("http://", "https://", "/", "./", "../")):
        return False
    if value.count("/") != 1:
        return False
    org, name = value.split("/", 1)
    if not org or not name or any(c in value for c in " :\\\n\t"):
        return False
    if name.endswith((".py", ".json", ".yaml", ".yml", ".txt", ".sh")):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value))


def scan(paths: list[Path], explicit: list[str]) -> dict[str, ModelRef]:
    found: dict[str, ModelRef] = {}
    for value in explicit:
        if valid_model_id(value):
            found.setdefault(value, ModelRef(value)).files.add("<command line>")
        else:
            print(f"warning: ignoring invalid model id: {value}", file=sys.stderr)
    for path in paths:
        try:
            text = path.read_text(errors="ignore")
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
            continue
        rel = str(path.relative_to(ROOT))
        for pattern in MODEL_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if valid_model_id(value):
                    found.setdefault(value, ModelRef(value)).files.add(rel)
    return dict(sorted(found.items()))


def model_dir(dest: Path, model_id: str) -> Path:
    return dest / model_id


def is_complete(path: Path) -> bool:
    if (path / MARKER).is_file():
        return True
    # Accept a manually staged model as present. A config without weights is
    # deliberately not enough because many tests instantiate a real model.
    has_config = (path / "config.json").is_file()
    has_weights = any(
        path.glob(pattern)
        for pattern in (
            "*.safetensors",
            "*.bin",
            "*.pt",
            "*.pth",
            "*.gguf",
            "*.index.json",
        )
    )
    return has_config and has_weights


def download(model_id: str, source: str, dest: Path, revision: str | None) -> None:
    target = model_dir(dest, model_id)
    if is_complete(target):
        print(f"SKIP {model_id} (already present: {target})")
        return
    target.mkdir(parents=True, exist_ok=True)
    print(f"DOWNLOAD {model_id} -> {target} [{source}]")
    if source == "huggingface":
        try:
            import huggingface_hub
        except ImportError as exc:
            raise SystemExit(
                "install huggingface_hub first: pip install huggingface_hub"
            ) from exc
        huggingface_hub.snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            revision=revision,
            resume_download=True,
        )
    else:
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise SystemExit(
                "install modelscope first: pip install modelscope"
            ) from exc
        kwargs = {"model_id": model_id, "local_dir": str(target)}
        if revision:
            kwargs["revision"] = revision
        snapshot_download(**kwargs)
    (target / MARKER).write_text("download completed\n")
    print(f"DONE {model_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--changed", action="store_true", help="scan files changed against origin/main"
    )
    scope.add_argument(
        "--all", action="store_true", help="scan the whole repository (default)"
    )
    parser.add_argument(
        "--model", action="append", default=[], help="model id, repeatable: org/name"
    )
    parser.add_argument(
        "--source", choices=("huggingface", "modelscope"), default="huggingface"
    )
    parser.add_argument("--revision", help="optional branch, tag, or commit")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--download", action="store_true", help="download after scanning"
    )
    parser.add_argument(
        "--list", action="store_true", help="only print discovered models"
    )
    args = parser.parse_args()

    refs = scan(candidate_files(args.changed, args.all), args.model)
    print(f"Found {len(refs)} model repositories:")
    for ref in refs.values():
        shown = sorted(ref.files)[:3]
        more = " ..." if len(ref.files) > 3 else ""
        print(f"- {ref.model_id}  ({', '.join(shown)}{more})")
    if args.list or not args.download:
        return 0
    args.dest.mkdir(parents=True, exist_ok=True)
    for ref in refs.values():
        download(ref.model_id, args.source, args.dest, args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
