#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fail when too large a fraction of a JUnit report was skipped.

A 4090 is SM89, so every test gated on sm90+/sm100 is skipped. pytest exits 0
in that case, which makes an all-skipped run indistinguishable from a passing
one. This guard makes that state visible.

Thresholds have to be calibrated per step against real hardware; run with
--warn-only until a baseline exists.
"""

from __future__ import annotations

import argparse
import glob
import sys
import xml.etree.ElementTree as ET


def read_counts(path: str) -> tuple[int, int, int, int]:
    """Return (tests, skipped, failures, errors) summed over all suites."""
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    tests = skipped = failures = errors = 0
    for suite in suites:
        tests += int(suite.get("tests", 0))
        skipped += int(suite.get("skipped", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
    return tests, skipped, failures, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="JUnit XML path (globs allowed)")
    parser.add_argument("--max", type=float, default=0.9, help="max skipped fraction")
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="report the ratio but always exit 0",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.report))
    if not paths:
        print(f"skip-ratio: no report at {args.report}; nothing to check")
        return 0

    tests = skipped = failures = errors = 0
    for path in paths:
        try:
            t, s, f, e = read_counts(path)
        except ET.ParseError as exc:
            print(f"skip-ratio: cannot parse {path}: {exc}")
            continue
        tests, skipped, failures, errors = (
            tests + t,
            skipped + s,
            failures + f,
            errors + e,
        )

    if tests == 0:
        print("skip-ratio: report collected 0 tests")
        return 0 if args.warn_only else 1

    ratio = skipped / tests
    detail = (
        f"skip-ratio: {skipped}/{tests} skipped ({ratio:.0%}), "
        f"{failures} failed, {errors} errored (threshold {args.max:.0%})"
    )

    if ratio <= args.max:
        print(detail)
        return 0

    print(f"{detail} -- EXCEEDS THRESHOLD")
    print(
        "This hardware likely cannot exercise these tests. Either move the step "
        "to step_denylist or raise its threshold once the baseline is known."
    )
    return 0 if args.warn_only else 1


if __name__ == "__main__":
    sys.exit(main())
