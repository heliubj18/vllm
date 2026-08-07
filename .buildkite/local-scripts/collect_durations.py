#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Accumulate measured step durations from Buildkite build history.

The cost tiers in ci_config_4090.yaml are inferred statically, by reading what
each test file does. That was necessary because upstream's timeout_in_minutes
measures its own network's model-download speed more than the work, and nothing
here had ever been timed. This script closes that gap from the other end: it
reads what steps actually took on this hardware.

Accumulating matters. A single build only times the steps that got far enough to
finish, and early on that was almost none - across builds #1-#17 exactly two GPU
steps ever completed. So this merges into an existing JSON store instead of
overwriting it, and every debug round adds whatever it managed to measure. The
markdown report is regenerated from the store, so it always reflects everything
collected so far rather than the last run.

Only `passed` jobs are recorded. A failed step's duration measures how long it
took to break, which is not a cost signal.

Not part of vLLM upstream CI.

Usage:

    export BUILDKITE_API_TOKEN=bkua_...
    .venv/bin/python .buildkite/local-scripts/collect_durations.py

    # or point it somewhere else
    .venv/bin/python .buildkite/local-scripts/collect_durations.py \
        --store ~/doc/vllm-4090-step-durations.json \
        --report ~/doc/vllm-4090-step-durations.md
"""

import argparse
import datetime
import json
import os
import pathlib
import statistics
import subprocess
import sys

ORG = "he-liu"
PIPELINE = "vllm"
API = f"https://api.buildkite.com/v2/organizations/{ORG}/pipelines/{PIPELINE}/builds"

# Buildkite timestamps. Fractional seconds are always present in practice, but
# the API is documented as RFC3339, so accept both.
STAMPS = ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ")

# The bootstrap steps on the mac agent. They are real jobs with real durations,
# but they say nothing about test cost and would crowd the report.
SKIP_LABELS = (
    ":wave: Hello world",
    ":mag: Environment",
    ":lock: Lint (ruff + typos + spdx)",
    ":pytest: Unit tests",
    ":pipeline: Generate 4090 pipeline",
    ":pipeline: upload",
    ":white_check_mark: Local CI green",
    "Hello world",
)


def parse_stamp(value: str) -> datetime.datetime | None:
    for fmt in STAMPS:
        try:
            return datetime.datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def fetch(token: str, pages: int) -> list[dict]:
    """Fetch build history via curl.

    curl rather than urllib because the system Python on the generating mac has
    no CA bundle configured and fails the TLS handshake against the API.
    """
    builds: list[dict] = []
    for page in range(1, pages + 1):
        url = f"{API}?per_page=30&page={page}"
        out = subprocess.run(
            ["curl", "-sS", "-H", f"Authorization: Bearer {token}", url],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if out.returncode != 0:
            print(f"curl failed on page {page}: {out.stderr[:200]}", file=sys.stderr)
            break
        try:
            batch = json.loads(out.stdout)
        except json.JSONDecodeError:
            print(f"page {page} was not JSON: {out.stdout[:200]}", file=sys.stderr)
            break
        if not batch:
            break
        builds.extend(batch)
    return builds


def harvest(builds: list[dict]) -> dict[str, dict]:
    """Pull (build number -> seconds) for every job that passed."""
    found: dict[str, dict] = {}
    for build in builds:
        for job in build.get("jobs") or []:
            label = (job.get("name") or "").strip()
            if not label or label in SKIP_LABELS:
                continue
            # A failed step's duration measures how long it took to break, not
            # what the work costs.
            if job.get("state") != "passed":
                continue
            started, finished = job.get("started_at"), job.get("finished_at")
            if not (started and finished):
                continue
            a, b = parse_stamp(started), parse_stamp(finished)
            if not (a and b):
                continue
            entry = found.setdefault(label, {})
            entry[str(build["number"])] = round((b - a).total_seconds(), 1)
    return found


def merge(store: dict, fresh: dict[str, dict]) -> tuple[int, int]:
    """Merge fresh measurements into the store. Returns (new labels, new runs)."""
    runs = store.setdefault("steps", {})
    new_labels = new_runs = 0
    for label, by_build in fresh.items():
        target = runs.setdefault(label, {})
        if not target:
            new_labels += 1
        for build, seconds in by_build.items():
            if build not in target:
                target[build] = seconds
                new_runs += 1
    store["updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return new_labels, new_runs


def human(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}m{total % 60:02d}s"


def report(store: dict) -> str:
    """Regenerate the markdown view from everything collected so far."""
    runs: dict[str, dict] = store.get("steps") or {}
    lines = [
        "# 4090 CI:各 step 实测耗时",
        "",
        "由 `.buildkite/local-scripts/collect_durations.py` 生成,**累积写入** ——",
        "每次 debug 后重跑该脚本,新测到的耗时并入,已有记录不覆盖。",
        "",
        "只记 `passed` 的 job:失败 step 的耗时衡量的是多久崩掉,不是工作量。",
        "",
        (f"最后更新:{store.get('updated', '-')}    已有 {len(runs)} 个 step 的数据"),
        "",
        "用途:替换 `ci_config_4090.yaml` 里静态推断的 cost tier,以及 `priority`",
        "的排序依据 —— 上游的 `timeout_in_minutes` 衡量的是它自己的下载带宽,不是",
        "这台机器上的真实开销。",
        "",
        "## 实测耗时",
        "",
        "| step | 次数 | 中位数 | 最短 | 最长 | 各次 (build:耗时) |",
        "|---|---|---|---|---|---|",
    ]

    def sort_key(item: tuple[str, dict]) -> float:
        return -statistics.median(item[1].values()) if item[1] else 0.0

    for label, by_build in sorted(runs.items(), key=sort_key):
        values = list(by_build.values())
        if not values:
            continue
        ordered = sorted(by_build.items(), key=lambda x: int(x[0]))
        detail = " ".join(f"#{b}:{human(s)}" for b, s in ordered)
        lines.append(
            f"| {label} | {len(values)} | {human(statistics.median(values))} "
            f"| {human(min(values))} | {human(max(values))} | {detail} |"
        )

    if not runs:
        lines.append("| (还没有数据) | | | | | |")

    total = sum(statistics.median(v.values()) for v in runs.values() if v)
    lines += [
        "",
        f"中位数合计:**{human(total)}** —— GPU step 串行(`gpu_slots: 1`),",
        "所以这个数字近似一次全量 build 的排队时间下界。",
        "",
        "## 已知但尚未测到的",
        "",
        "表里缺的 step 不是不存在,是还没有一次成功跑完过。生成器当前选中 25 个",
        "step,凑齐需要一轮完整的绿。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    home = pathlib.Path.home()
    parser.add_argument(
        "--store",
        type=pathlib.Path,
        default=home / "doc" / "vllm-4090-step-durations.json",
        help="JSON store that accumulates across runs",
    )
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        default=home / "doc" / "vllm-4090-step-durations.md",
        help="markdown view, regenerated from the store each run",
    )
    parser.add_argument("--pages", type=int, default=2, help="pages of 30 builds")
    args = parser.parse_args()

    token = os.environ.get("BUILDKITE_API_TOKEN")
    if not token:
        print("BUILDKITE_API_TOKEN is not set", file=sys.stderr)
        return 1

    builds = fetch(token, args.pages)
    if not builds:
        print("no builds fetched", file=sys.stderr)
        return 1

    store = {}
    if args.store.exists():
        store = json.loads(args.store.read_text())

    fresh = harvest(builds)
    new_labels, new_runs = merge(store, fresh)

    args.store.parent.mkdir(parents=True, exist_ok=True)
    args.store.write_text(json.dumps(store, indent=2, ensure_ascii=False) + "\n")
    args.report.write_text(report(store))

    measured = len(store.get("steps") or {})
    print(f"扫了 {len(builds)} 个 build")
    print(f"新增 {new_labels} 个 step、{new_runs} 次测量")
    print(f"累计 {measured} 个 step 有数据")
    print(f"  store  {args.store}")
    print(f"  report {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
