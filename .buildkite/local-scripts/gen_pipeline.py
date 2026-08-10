#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate a Buildkite pipeline for a self-hosted agent from upstream steps.

Reads the step definitions in .buildkite/test_areas/ (upstream files, never
modified) and emits a pipeline containing only the steps a given agent can run.

All hardware- and platform-specific choices live in the config file, not here:
GPU count, blocked architectures, cost thresholds and the must-run allowlist are
config keys. Adding an agent means adding a ci_config_<platform>.yaml and
pointing --config at it; this script should not need changes.

Usage:
    python3 .buildkite/local-scripts/gen_pipeline.py --config <cfg> \\
        | buildkite-agent pipeline upload
    python3 .buildkite/local-scripts/gen_pipeline.py --config <cfg> --dry-run

This is NOT part of vLLM upstream CI.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import regex as re
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / ".buildkite" / "ci_config_4090.yaml"
DEFAULT_TEST_AREAS = REPO_ROOT / ".buildkite" / "test_areas"

# Where a step runs when it declares no working_dir. Most upstream steps say
# /vllm-workspace/tests explicitly and the rest still name paths relative to it,
# so both the emitted workdir and the static target resolution use this.
DEFAULT_WORKING_DIR = "/vllm-workspace/tests"


# --------------------------------------------------------------------------
# Upstream step model
# --------------------------------------------------------------------------


@dataclass
class Step:
    """One step as declared in an upstream test_areas YAML file."""

    label: str
    group: str = ""
    key: str | None = None
    working_dir: str | None = None
    commands: list[str] = field(default_factory=list)
    device: str | None = None
    num_devices: int = 1
    num_nodes: int | None = None
    source_file_dependencies: list[str] = field(default_factory=list)
    timeout_in_minutes: int | None = None
    parallelism: int | None = None
    optional: bool = False
    no_gpu: bool = False
    env: dict[str, str] = field(default_factory=dict)
    # Filled in by classify(); not part of the upstream YAML.
    cost: Cost | None = None

    @classmethod
    def from_yaml(cls, data: dict[str, Any], group: str) -> Step:
        return cls(
            label=str(data.get("label", "")).strip(),
            group=group,
            key=data.get("key"),
            working_dir=data.get("working_dir"),
            commands=list(data.get("commands") or []),
            device=data.get("device"),
            num_devices=int(data.get("num_devices") or 1),
            num_nodes=data.get("num_nodes"),
            source_file_dependencies=list(data.get("source_file_dependencies") or []),
            timeout_in_minutes=data.get("timeout_in_minutes"),
            parallelism=data.get("parallelism"),
            optional=bool(data.get("optional", False)),
            no_gpu=bool(data.get("no_gpu", False)),
            env=dict(data.get("env") or {}),
        )

    @property
    def slug(self) -> str:
        """Stable key usable as a Buildkite step key."""
        if self.key:
            return self.key
        return re.sub(r"[^a-zA-Z0-9_-]+", "-", self.label.lower()).strip("-")


def load_steps(test_areas: Path) -> list[Step]:
    steps: list[Step] = []
    for path in sorted(test_areas.glob("*.yaml")):
        with open(path) as handle:
            data = yaml.safe_load(handle) or {}
        group = data.get("group", path.stem)
        for raw in data.get("steps") or []:
            # `mirror` holds AMD-only variants; drop it by simply not reading it.
            steps.append(Step.from_yaml(raw, group))
    return steps


def load_config(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    _resolve_image(config)
    return config


def _resolve_image(config: dict[str, Any]) -> None:
    """Apply the `images.use` preset over the top-level `image`/`always_pull`.

    Swapping the image swaps what is under test - the test code comes from the
    checkout either way - so this is the seam for running the same suite against
    an internal build instead of upstream's.

    The arch check is the part worth having. An image compiled without this box's
    compute capability still starts, still imports vllm, and then fails somewhere
    inside a kernel with a message about missing symbols or no kernel image. That
    reads like a bug in the code under test and is not one. Refusing to generate a
    pipeline is the cheaper failure.
    """
    images = config.get("images") or {}
    presets = images.get("presets") or {}
    name = images.get("use")
    if not name:
        return
    if name not in presets:
        raise SystemExit(
            f"images.use is {name!r}, which is not in images.presets "
            f"({', '.join(sorted(presets)) or 'empty'})"
        )
    preset = presets[name] or {}
    if preset.get("image"):
        config["image"] = preset["image"]
    if "always_pull" in preset:
        config.setdefault("docker", {})["always_pull"] = bool(preset["always_pull"])

    # `arch_list` is what the image reports from torch.cuda.get_arch_list(); it
    # has to be recorded by whoever adds the preset, since we cannot inspect a
    # not-yet-pulled image at generation time on the mac agent.
    arch = (config.get("hardware") or {}).get("arch")
    declared = preset.get("arch_list")
    if arch and declared is not None and not _arch_is_covered(arch, declared):
        raise SystemExit(
            f"image preset {name!r} declares arch_list {declared or '[]'}, which "
            f"does not cover this box's {arch}.\n"
            f"  image: {preset.get('image')}\n"
            f"  note : {preset.get('note', '-')}\n"
            "Rebuild with this architecture in torch_cuda_arch_list, or pick "
            "another preset. Running anyway produces kernel failures that look "
            "like defects in the code under test."
        )


def _preset_suffix(config: dict[str, Any]) -> str:
    """Name the active preset in the summary, so a switch is visible at a glance."""
    use = (config.get("images") or {}).get("use")
    return f"   (preset: {use})" if use else ""


def _arch_is_covered(arch: str, declared: list[str]) -> bool:
    """Whether `declared` can run `arch` (e.g. sm89), natively or via PTX.

    A cubin only runs on its exact target, so sm_89 needs sm_89. PTX is
    forward-compatible within a major generation: sm_86 PTX runs on sm_89
    because both are 8.x, but 10.0+PTX cannot lower to 8.9.

    >>> _arch_is_covered("sm89", ["sm_86", "sm_90", "sm_120"])  # upstream image
    True
    >>> _arch_is_covered("sm89", ["sm_90", "sm_100", "sm_103"])  # novita overlay
    False
    >>> _arch_is_covered("sm89", [])
    False
    >>> _arch_is_covered("sm89", ["sm_89"])
    True
    >>> _arch_is_covered("sm90", ["sm_100"])
    False
    """
    if arch in declared:
        return True
    major = arch.removeprefix("sm").lstrip("_")[:1]
    return any(d.removeprefix("sm_")[:1] == major and d != arch for d in declared)


# --------------------------------------------------------------------------
# Static cost estimation
# --------------------------------------------------------------------------
#
# `timeout_in_minutes` is a poor cost proxy here: upstream sizes it to include
# cold model downloads over its own network, so it mostly measures bandwidth.
# With weights pre-staged in a local HF cache the real cost is dominated by how
# a test obtains a model, which is visible in the test source.

MODEL_RE = re.compile(r"""["']([A-Za-z0-9][\w.-]*/[\w.-]+)["']""")
_NOT_A_MODEL = (
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".so",
    ".safetensors",
    ".jinja",
    ".md",
    ".csv",
    ".sh",
    ".cu",
    ".h",
    ".toml",
)
_SERVER_RE = re.compile(r"RemoteOpenAIServer")
_ENGINE_RE = re.compile(r"\bLLM\(|AsyncLLM|vllm_runner|hf_runner")
_COMPILE_RE = re.compile(r"torch\.compile|cudagraph|CUDAGraph|CompilationConfig")
_MULTIGPU_RE = re.compile(r"tensor_parallel_size\s*=\s*[2-9]|distributed_executor")
# A step that invokes a shell script may load models inside it, out of sight.
_SHELLS_OUT_RE = re.compile(r"\bbash\s|\.sh\b|\bsh\s")

TIER_ORDER = ["T0", "T1", "T2", "T3"]
TIER_UNKNOWN = "TU"


@dataclass
class Cost:
    """Static cost profile of a step, derived from the files it collects."""

    tier: str
    detail: str
    files: int = 0
    server_files: int = 0
    engine_files: int = 0
    models: int = 0
    compile_files: int = 0
    multigpu_files: int = 0

    @property
    def spawns_server(self) -> bool:
        """Whether any collected file stands up a `vllm serve` process.

        The only case where upstream's timeout still carries signal: server
        startup and health polling are real per-test cost that pre-staged
        weights do not remove, and are not countable from the source.
        """
        return self.server_files > 0


def _resolve_targets(step: Step) -> list[str]:
    """Test files a step collects, resolved against its working_dir.

    Upstream commands name paths relative to the step's working_dir, which
    defaults to /vllm-workspace/tests. Returns [] when the targets cannot be
    resolved statically (shell wrappers, generated arg lists), which the caller
    treats as "unknown" rather than "cheap".
    """
    work = (step.working_dir or DEFAULT_WORKING_DIR).replace("/vllm-workspace", "")
    prefix = work.strip("/")
    collected: set[str] = set()

    for command in step.commands:
        if "pytest" not in command:
            continue
        marker = re.search(r"-m\s+['\"]([^'\"]+)['\"]", command)
        ignored = set(re.findall(r"--ignore[= ]([^\s]+)", command))
        picked: set[str] = set()

        for token in re.split(r"[\s;&|]+", command):
            token = token.strip("'\"").rstrip(".,:")
            if not token or token.startswith("-") or "=" in token:
                continue
            for candidate in {os.path.join(prefix, token) if prefix else token, token}:
                path = REPO_ROOT / os.path.normpath(candidate)
                if not str(path).startswith(str(REPO_ROOT / "tests")):
                    continue
                if path.is_dir():
                    picked.update(str(p) for p in path.rglob("test_*.py"))
                elif path.is_file() and path.name.startswith("test_"):
                    picked.add(str(path))

        for entry in ignored:
            for candidate in {os.path.join(prefix, entry) if prefix else entry, entry}:
                picked.discard(str(REPO_ROOT / os.path.normpath(candidate)))

        # A marker expression selects a fraction of the directory; approximate
        # it by keeping only files that mention the marker name.
        if marker:
            name = marker.group(1)
            picked = {p for p in picked if name in _read(p)}

        collected.update(picked)

    return sorted(collected)


_SOURCE_CACHE: dict[str, str] = {}


def _read(path: str) -> str:
    """Read a test file once; the same file is collected by several steps."""
    if path not in _SOURCE_CACHE:
        try:
            _SOURCE_CACHE[path] = Path(path).read_text(errors="ignore")
        except OSError:
            _SOURCE_CACHE[path] = ""
    return _SOURCE_CACHE[path]


def estimate_cost(step: Step, config: dict[str, Any]) -> Cost:
    """Assign a cost tier from what the step's test files do.

    T0  no model touched at all, or no pytest to run
    T1  in-process tests over a handful of models
    T2  loads weights into VRAM, or compiles in most files
    T3  spawns `vllm serve`, or sweeps many distinct models
    TU  targets not statically resolvable; caller falls back to timeout

    The model count is an upper bound: a repo id passed to `create_vllm_config`
    only pulls HF metadata, not weights. It is used as a magnitude signal, so
    over-counting biases toward MANUAL, which is the safe direction.
    """
    rules = config.get("cost") or {}
    many_models = int(rules.get("many_models") or 20)

    if not any("pytest" in command for command in step.commands):
        # A wrapper script can still drive pytest and load models internally
        # (weight_loading does), so only a step that shells out to nothing at
        # all is genuinely free. Anything else stays unknown.
        if any(_SHELLS_OUT_RE.search(command) for command in step.commands):
            return Cost(TIER_UNKNOWN, "runs a wrapper script, cost opaque")
        return Cost("T0", "no pytest")

    files = _resolve_targets(step)
    if not files:
        return Cost(TIER_UNKNOWN, "targets not statically resolvable")

    models: set[str] = set()
    server = engine = plain = compiles = multigpu = 0
    for path in files:
        source = _read(path)
        for match in MODEL_RE.finditer(source):
            name = match.group(1)
            if name.count("/") == 1 and not name.endswith(_NOT_A_MODEL):
                models.add(name)
        if _SERVER_RE.search(source):
            server += 1
        elif _ENGINE_RE.search(source):
            engine += 1
        else:
            plain += 1
        if _COMPILE_RE.search(source):
            compiles += 1
        if _MULTIGPU_RE.search(source):
            multigpu += 1

    profile = dict(
        files=len(files),
        server_files=server,
        engine_files=engine,
        models=len(models),
        compile_files=compiles,
        multigpu_files=multigpu,
    )

    if server:
        return Cost("T3", f"{server} file(s) spawn a server", **profile)
    if len(models) > many_models:
        return Cost("T3", f"{len(models)} distinct models", **profile)
    if engine:
        return Cost("T2", f"{engine} file(s) load an engine", **profile)
    if compiles * 2 >= len(files):
        return Cost("T2", f"{compiles}/{len(files)} files compile", **profile)
    if not models:
        return Cost("T0", f"{plain} in-process file(s), no model", **profile)
    return Cost("T1", f"{plain} in-process file(s), {len(models)} models", **profile)


# --------------------------------------------------------------------------
# Changed-file detection
# --------------------------------------------------------------------------


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=REPO_ROOT
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def changed_files() -> tuple[list[str], str]:
    """Return (changed paths, human-readable description of the comparison)."""
    base_branch = os.environ.get("BUILDKITE_PULL_REQUEST_BASE_BRANCH") or "main"
    if os.environ.get("BUILDKITE_PULL_REQUEST", "false") == "false":
        base_branch = "main"

    merge_base = os.environ.get("MERGE_BASE_COMMIT") or _git(
        "merge-base", f"origin/{base_branch}", "HEAD"
    )
    if not merge_base:
        return [], "no merge base found (treating as run-all)"

    out = _git("diff", "--name-only", "--diff-filter=ACMDR", merge_base, "HEAD")
    files = [line for line in out.splitlines() if line.strip()]
    return files, f"{merge_base[:8]}..HEAD ({len(files)} files)"


def matches_run_all(files: list[str], config: dict[str, Any]) -> str | None:
    """Return the file that forces a full run, or None."""
    include = config.get("run_all_patterns") or []
    exclude = config.get("run_all_exclude_patterns") or []
    for path in files:
        if not any(path.startswith(p) for p in include):
            continue
        if any(path.startswith(p) for p in exclude):
            continue
        return path
    return None


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------

AUTO = "auto"  # runs on every build
MANUAL = "manual"  # emitted behind a Buildkite block step


@dataclass
class Verdict:
    step: Step
    mode: str | None  # AUTO, MANUAL, or None when excluded
    reason: str


def classify(
    step: Step,
    config: dict[str, Any],
    files: list[str],
    run_all: bool,
) -> Verdict:
    """Decide whether and how a step runs. Layers are applied in order."""
    hardware = config.get("hardware") or {}
    rules = config.get("filter") or {}

    # ---- layer 1: hardware capability ----
    if not step.commands:
        return Verdict(step, None, "no commands")

    allowlist = rules.get("step_allowlist") or []
    if allowlist and step.slug not in allowlist:
        return Verdict(step, None, "not in step_allowlist")

    if step.slug in (rules.get("step_denylist") or []):
        return Verdict(step, None, "in step_denylist")

    if step.num_nodes:
        return Verdict(step, None, f"needs {step.num_nodes} nodes")

    gpu_count = int(hardware.get("gpu_count") or 1)
    if step.num_devices > gpu_count:
        return Verdict(step, None, f"needs {step.num_devices} GPUs, have {gpu_count}")

    device = (step.device or "").strip()
    if device:
        if device in (rules.get("arch_blocklist") or []):
            return Verdict(step, None, f"device {device} not available")
        for prefix in rules.get("arch_blocklist_prefixes") or []:
            if device.startswith(prefix):
                return Verdict(step, None, f"device {device} matches '{prefix}*'")

    # ---- layer 2: cost ----
    # Tier comes from what the test files do (see estimate_cost). Upstream's
    # timeout is consulted only for steps that stand up a server, where startup
    # and readiness polling dominate and are not visible in the source. For
    # everything else the timeout mostly reflects cold model downloads on
    # upstream's network, which does not apply to a pre-staged local cache.
    cost_rules = config.get("cost") or {}
    auto_max_tier = str(cost_rules.get("auto_max_tier") or "T2")
    cost = estimate_cost(step, config)
    step.cost = cost

    must_run = cost_rules.get("must_run") or []
    if step.slug in must_run:
        return Verdict(step, AUTO, f"{cost.tier}; in cost.must_run")

    if step.optional:
        return Verdict(step, MANUAL, "upstream marks it optional")

    over_tier = cost.tier in TIER_ORDER and TIER_ORDER.index(
        cost.tier
    ) > TIER_ORDER.index(auto_max_tier)

    if cost.spawns_server:
        # Timeout can only narrow the auto set here, never widen it past the
        # tier ceiling: upstream's number is unreliable in both directions.
        # Extract Hidden States claims 20min while sweeping 70+ models.
        timeout = step.timeout_in_minutes or 0
        auto_max = int(rules.get("auto_max_minutes") or 20)
        if timeout > auto_max:
            mode = MANUAL
            reason = f"{cost.detail}; {timeout}min > auto_max_minutes={auto_max}"
        elif over_tier:
            mode = MANUAL
            reason = f"{cost.tier} ({cost.detail}) above auto_max_tier={auto_max_tier}"
        else:
            mode, reason = AUTO, f"{cost.detail}; {timeout}min"
    elif cost.tier == TIER_UNKNOWN:
        mode, reason = MANUAL, cost.detail
    elif over_tier:
        mode = MANUAL
        reason = f"{cost.tier} ({cost.detail}) above auto_max_tier={auto_max_tier}"
    else:
        mode, reason = AUTO, f"{cost.tier} ({cost.detail})"

    # ---- layer 3: changed files ----
    if run_all:
        return Verdict(step, mode, f"{reason}; run-all")

    deps = step.source_file_dependencies
    hit = next(
        (f for f in files for d in deps if d and f.startswith(d)),
        None,
    )
    if hit:
        return Verdict(step, mode, f"{reason}; triggered by {hit}")

    if rules.get("block_unselected", True):
        return Verdict(step, MANUAL, f"{reason}; not triggered by this diff")
    return Verdict(step, None, "not triggered by this diff")


# --------------------------------------------------------------------------
# Buildkite step emission
# --------------------------------------------------------------------------


def _is_cpu_only(step: Step) -> bool:
    """Whether a step needs no GPU, so the container should not reserve one."""
    return step.no_gpu or (step.device or "").startswith("cpu")


def _runs_pytest(step: Step) -> bool:
    """Whether any command invokes pytest, and so produces a JUnit report."""
    return any("pytest" in command for command in step.commands)


def _junit_path(step: Step, report: dict[str, Any]) -> str:
    junit_dir = report.get("junit_dir") or "/vllm-workspace/test-reports"
    return f"{junit_dir}/{step.slug}.xml"


CHECKOUT_STAGING = "/checkout-ro"


def _copy_commands(config: dict[str, Any]) -> list[str]:
    """Copy the checkout mounts into the workspace before any test runs.

    The agent keeps one working directory per agent instance - the path is
    <build-path>/<agent-name>/<org>/<pipeline> with no build number - so every
    build it handles reuses the same checkout, and each job starts by cleaning it
    and checking out its own build's commit. The volume mounts are live views of
    that directory, not snapshots, so a step running while another build's job
    starts sees its own tests/ swapped underneath it. Nothing fails loudly; the
    step just runs against another PR's source and reports under its own commit.

    Copying decouples the two. It costs 0.25s for 22M of tests/, against steps
    that run for tens of minutes.

    Directories are copied with a trailing `/.` so the contents land in an
    existing destination rather than nesting (cp -r src dst puts src *inside* dst
    when dst exists, which would give /vllm-workspace/tests/tests).
    """
    docker = config.get("docker") or {}
    names = docker.get("copy_from_checkout")
    if not names:
        return []

    out = ["mkdir -p /vllm-workspace"]
    for name in names:
        src = f"{CHECKOUT_STAGING}/{name}"
        dst = f"/vllm-workspace/{name}"
        # A file (pyproject.toml) and a directory need different forms, and the
        # step cannot know which it is at generation time - decide in the shell.
        out.append(
            f'if [ -d "{src}" ]; then mkdir -p "{dst}" && cp -a "{src}/." "{dst}/"; '
            f'else cp -a "{src}" "{dst}"; fi'
        )
    return out


def _select_commands(step: Step, config: dict[str, Any]) -> list[str]:
    """Drop commands this box cannot run, keeping the rest of the step.

    A Buildkite step is atomic: one failing command fails the whole step, and
    under fail_fast that cancels everything queued behind it. Upstream never
    needs this because each step gets a machine sized for it, but here a single
    step can mix commands that work with commands that cannot:

      v1-others-cpu   `-m cpu_test v1/core` wants world size 2 and
                      `-m cpu_test v1/kv_connector/unit` wants 4 (tp2 x pp2),
                      while the other five commands need no device. Upstream runs
                      the step on image-build-cpu, where ParallelConfig's
                      world-size check is inert; on a CUDA image it is live.

    Matching is a plain substring against the command text, so an entry stays
    readable and survives the `$$` unescaping that happens later. A pattern that
    matches nothing is reported by --verbose rather than failing the build: the
    upstream command may simply have been reworded, and a stale entry that
    silently keeps a command is safer than one that silently drops it.
    """
    drops = (config.get("filter") or {}).get("command_denylist") or {}
    patterns = drops.get(step.key) or []
    if not patterns:
        return list(step.commands)
    return [c for c in step.commands if not any(p in c for p in patterns)]


def _deselects(step: Step, config: dict[str, Any]) -> list[str]:
    """pytest node ids to --deselect for this step, from filter.test_denylist."""
    drops = (config.get("filter") or {}).get("test_denylist") or {}
    return list(drops.get(step.key) or [])


def _wrap_commands(step: Step, config: dict[str, Any]) -> list[str]:
    """Add JUnit reporting and the skip-ratio guard around upstream commands.

    Upstream commands are never rewritten: pytest gets its options through
    PYTEST_ADDOPTS so the report lands in a known place without touching any
    command string.
    """
    prologue = _copy_commands(config)
    commands = _select_commands(step, config)

    # Steps that never call pytest (static audits, cargo, shell checks) get no
    # report, so wrapping them would only add a spurious skip-ratio check. They
    # still need the copy: their commands read from the checkout too. Judged on
    # the surviving commands, so a step whose only pytest call was filtered out
    # does not get a report path pointing at a file nothing will write.
    if not any("pytest" in c for c in commands):
        return prologue + list(commands)

    report = config.get("report") or {}
    junit_dir = report.get("junit_dir") or "/vllm-workspace/test-reports"
    junit = _junit_path(step, report)
    # Absolute so it resolves regardless of the step's working_dir.
    checker = report.get(
        "checker_path", "/vllm-workspace/.buildkite/local-scripts/check_skip_ratio.py"
    )

    out = [*prologue, f"mkdir -p {junit_dir}"]
    # We drop upstream's `parallelism`, so Buildkite never sets the two shard
    # variables. Commands that pass them to pytest would expand to an empty
    # `--shard-id= --num-shards=` and fail at collection, so define them as the
    # single-shard case. Harmless for steps that do not shard.
    out.append("export BUILDKITE_PARALLEL_JOB=${BUILDKITE_PARALLEL_JOB:-0}")
    out.append("export BUILDKITE_PARALLEL_JOB_COUNT=${BUILDKITE_PARALLEL_JOB_COUNT:-1}")
    # PYTEST_ADDOPTS applies to every pytest process the step spawns, including
    # the ones launched from upstream's shell wrappers.
    addopts = [f"--junitxml={junit}", "-o junit_family=xunit2"]
    # Individual tests dropped with --deselect. This is the finer-grained sibling
    # of command_denylist: that one loses every test in a command, which for a
    # command like `-m cpu_test multimodal` means giving up a few hundred passing
    # cases to skip one. --deselect goes through PYTEST_ADDOPTS so no command
    # string is rewritten, and a stale entry is inert rather than fatal - pytest
    # ignores a --deselect target that does not exist.
    addopts += [f"--deselect {t}" for t in _deselects(step, config)]
    out.append(f'export PYTEST_ADDOPTS="{" ".join(addopts)}"')
    out.extend(_unescape_dollars(c) for c in commands)

    enforce = bool(report.get("skip_ratio_enforce", False))
    threshold = report.get("skip_ratio_default", 0.9)
    guard = f"python3 {checker} {junit} --max {threshold}"
    if not enforce:
        guard += " --warn-only"
    # `|| true` keeps a warn-only guard from masking the real exit status.
    out.append(guard if enforce else f"{guard} || true")
    return out


def _unescape_dollars(command: str) -> str:
    """Undo the `$$` escaping upstream commands rely on the uploader to resolve.

    Upstream writes `--shard-id=$$BUILDKITE_PARALLEL_JOB`, counting on
    `buildkite-agent pipeline upload` to collapse `$$` to a single `$` during
    interpolation. We upload with --no-interpolation - required so
    $BUILDKITE_BUILD_CHECKOUT_PATH resolves on the test agent rather than the
    mac that generates the pipeline - which means nothing performs that
    collapse. bash then reads `$$` as its own PID and the rest as literal text:

        pytest: error: argument --shard-id:
                invalid positive_int value: '7BUILDKITE_PARALLEL_JOB'

    So do the collapse here, where the two requirements meet. This is the one
    place a command string is rewritten; it restores upstream's intent rather
    than changing it.
    """
    return command.replace("$$", "$")


def _gpus_for(step: Step, config: dict[str, Any]) -> str:
    """Which devices this step's container gets.

    Upstream declares how many devices a step needs (`num_devices`), and until
    now every step was handed the same string regardless - so a step needing one
    card still had both attached, and two such steps could never share the box.
    Sizing the allocation per step is what makes running two of them at once
    possible at all.

    The value must be decided here, at generation time. The docker plugin passes
    it through verbatim - `args+=("--gpus" "${BUILDKITE_PLUGIN_DOCKER_GPUS:-}")`
    in hooks/command:381 - and only volume paths get `eval echo`, so a `$VAR` in
    this field reaches docker as a literal and the container fails to start. The
    same constraint rules out giving each shard of a `parallelism` step its own
    card: shards share one step definition, so they would share one device string.

    Falls back to `gpus` when no per-count entry matches, which keeps configs that
    never set `gpus_by_devices` behaving exactly as before.
    """
    docker = config.get("docker") or {}
    default = docker.get("gpus", "all")

    by_count = docker.get("gpus_by_devices") or {}
    if not by_count:
        return default
    # CPU-only steps still need a visible device (importing vllm loads a .so that
    # links libcuda.so.1), but never more than one, so they take the 1-device
    # allocation rather than the default.
    wanted = 1 if _is_cpu_only(step) else step.num_devices
    # A step can need more cards than its upstream marking implies, because
    # upstream's marking encodes which image it runs on rather than what its cases
    # ask for. v1-others-cpu is the case in point: upstream runs it on a CPU image
    # where ParallelConfig does not check GPU count, so a scheduler unit test can
    # declare world size 2 while touching no device. On a CUDA image that check is
    # live and the case fails.
    wanted = int((docker.get("devices_by_step") or {}).get(step.key, wanted))
    return by_count.get(wanted, default)


def _priority(step: Step, config: dict[str, Any]) -> int:
    """Dispatch order within the concurrency queue. Higher goes first.

    With gpu_slots at 1 the GPU steps form a single queue shared by every open
    PR, and Buildkite drains it in dispatch order. So one PR's 100-minute kernel
    sweep can sit ahead of another PR's 4-minute step, and the second PR waits
    the full 100 minutes for a verdict it could have had immediately. Priority is
    Buildkite's own answer to this: it reorders what is already queued, without
    changing what runs or how much of it.

    Ordering is by cost, cheapest first, and deliberately NOT by importance.

    The tempting rule - put cost.must_run at the front because it is the coverage
    that must not regress - makes things worse here. Those three kernel steps
    carry upstream timeouts of 100, 120 and 130 minutes, and Kernels Attention
    was measured at 100 minutes on this box. Draining them first means a second
    PR waits some five hours before its own steps are even dispatched. must_run
    guarantees a step *runs*; it says nothing about it running early.

    So the tiers are:

      short      upstream timeout at or under short_max_minutes. A cheap step
                 cannot delay the queue much, and finishing it early gives
                 whoever is waiting a real verdict instead of a spinner.
      default    everything else, in the order Buildkite already had.

    Upstream's timeout_in_minutes is a poor absolute cost model - it is sized for
    cold model downloads on upstream's network, which pre-staged weights remove,
    which is why cost tiers exist at all. For *ordering* it is good enough: it is
    monotonic with the work, and a wrong guess only changes queue order. Once
    real per-step durations are collected from the JUnit reports, they should
    replace it here.
    """
    rules = config.get("priority") or {}
    if not rules.get("enabled", True):
        return 0

    short_max = rules.get("short_max_minutes")
    # Per shard upstream, so scale by parallelism to get the step's own cost -
    # the same correction emit_step applies to the timeout it emits.
    timeout = step.timeout_in_minutes
    if timeout:
        timeout *= step.parallelism or 1
    if short_max and timeout and timeout <= int(short_max):
        return int(rules.get("short") or 5)

    return 0


def _concurrency(step: Step, config: dict[str, Any]) -> tuple[str | None, int]:
    """Return (concurrency_group, limit) for a step, or (None, 0) to leave it free.

    Two groups, split by what the step actually contends for.

    VRAM is the scarce resource and only GPU steps consume it, so those share one
    group capped at `gpu_slots` - across builds, since concurrency groups are
    organization-wide. That is what keeps several open PRs from stacking GPU work
    on one box.

    CPU-only steps get their own group. They are handed the devices too (see
    emit_step: importing vllm needs libcuda.so.1, so a GPU-less container fails
    at import) but allocate no VRAM, measured at 1 MiB while running. So they can
    overlap GPU work, and each other, bounded by `cpu_slots` for host RAM and
    cores rather than memory on the card.

    Buildkite offers no mutual exclusion *between* groups, so the two limits add
    up rather than capping a total: gpu_slots 1 + cpu_slots 2 permits three
    containers. That is deliberate here - the third and fourth containers are not
    competing for the thing that runs out.
    """
    rules = config.get("concurrency") or {}
    if not rules.get("enabled", True):
        return None, 0

    hardware = config.get("hardware") or {}
    queue = hardware.get("queue_gpu") or "default"
    prefix = rules.get("group_prefix") or "vllm-ci"

    if _is_cpu_only(step):
        limit = int(rules.get("cpu_slots") or 2)
        return f"{prefix}/{queue}/cpu", limit

    # Steps needing a single card can be given their own group and run several at
    # a time, once gpus_by_devices sizes their allocation down to one device -
    # otherwise they would each hold every card and "several at a time" would just
    # be VRAM contention.
    #
    # This does NOT compose into a safe total. Buildkite counts jobs per group and
    # has no mutual exclusion between groups, so a 1-device step and a 2-device
    # step can run together: three devices' worth of demand on two cards, with the
    # 1-device step's card overlapping one of the pair. Cheap kernel tests coexist
    # there (measured at 460 MiB), a step that loads weights would not. So
    # gpu1_slots stays absent by default, and turning it on is a statement about
    # the steps involved.
    if step.num_devices <= 1 and rules.get("gpu1_slots") is not None:
        return f"{prefix}/{queue}/gpu1", int(rules["gpu1_slots"])

    # One multi-device step at a time. Raising this only makes sense if the steps'
    # num_devices sum to no more than the box's GPU count, which is not something
    # this generator can guarantee, so it stays a manual choice.
    limit = int(rules.get("gpu_slots") or 1)
    return f"{prefix}/{queue}/gpu", limit


def emit_step(step: Step, mode: str, config: dict[str, Any]) -> dict[str, Any]:
    hardware = config.get("hardware") or {}
    rules = config.get("filter") or {}
    report = config.get("report") or {}

    multiplier = float(rules.get("timeout_multiplier") or 1.0)
    timeout = step.timeout_in_minutes
    if timeout:
        # Upstream's timeout is per shard, not per step. A step with
        # `parallelism: 5` is five jobs upstream, each given this many minutes for
        # a fifth of the tests; we drop parallelism and run all of them in one
        # job, so the whole step needs the sum.
        #
        # Missing this timed out Kernels MoE in build #17: upstream 50 minutes
        # times parallelism 5 is 250 minutes of work, we allowed 50 x 2.0 = 100,
        # and the agent killed it at 101m01s with exit -1. Kernels Attention and
        # Quantization survived only because their parallelism is 2, which the
        # 2.0 multiplier happened to cover.
        timeout *= step.parallelism or 1
        timeout = max(1, int(round(timeout * multiplier)))

    docker = config.get("docker") or {}
    plugin: dict[str, Any] = {
        "image": config.get("image"),
        "always-pull": bool(docker.get("always_pull", False)),
        "propagate-environment": True,
        # No `pid` key on purpose. Docker's default is already a private PID
        # namespace (verified: a container sees 4 processes and is PID 1), which
        # is the isolation the upstream PD scripts need for their unfiltered
        # `pkill -9 -f "vllm serve"`. Asking for it explicitly is worse than
        # useless: the plugin passes the value straight through, and
        # `--pid private` is not valid docker syntax, so every job exits 125
        # with "invalid PID mode" before the container starts.
        # vLLM's multiprocessing needs more shared memory than docker's 64MB.
        "shm-size": docker.get("shm_size", "8gb"),
        # The serving images set ENTRYPOINT to `vllm serve`, which swallows the
        # step's commands as its own arguments and exits 125 before anything
        # runs. Override it so the commands reach a shell.
        "entrypoint": docker.get("entrypoint", "bash"),
        # Must accompany `entrypoint`. The plugin only injects its default
        # shell (/bin/sh -e -c) when no entrypoint is set, so overriding the
        # entrypoint alone hands the whole multi-line script to bash as a
        # filename: "bash: mkdir -p /vllm-workspace/test-reports ..." and exit
        # 127. Upstream commands assume bash, so -e -c under bash it is.
        "shell": list(docker.get("shell") or ["-e", "-c"]),
    }
    # The plugin mounts the checkout over `workdir` by default. That puts the
    # checkout's own vllm/ source tree on sys.path, where it shadows the
    # compiled package in the image, and kernel tests fail with
    # "_OpNamespace '_C' object has no attribute gelu_fast". Configs that mount
    # only tests/ themselves must turn it off.
    if "mount_checkout" in docker:
        plugin["mount-checkout"] = bool(docker["mount_checkout"])
    # Volumes may reference agent variables such as $BUILDKITE_BUILD_CHECKOUT_PATH,
    # which must resolve on the test agent, so the upload runs with
    # --no-interpolation. The plugin only expands variables inside volume paths
    # when this is on (it defaults to off), and without it docker receives the
    # literal string "$BUILDKITE_BUILD_CHECKOUT_PATH/tests" as a host path.
    if docker.get("expand_volume_vars", True):
        plugin["expand-volume-vars"] = True
    if docker.get("volumes"):
        plugin["volumes"] = list(docker["volumes"])
    # Steps that omit working_dir still name paths relative to tests/, e.g.
    # `pytest -m 'cpu_test' v1/core`. Upstream gets away with it because its
    # image has the tests baked in and the plugin's own checkout mount lands on
    # workdir. Here mount-checkout is off, so an unset workdir leaves pytest in
    # /vllm-workspace with "file or directory not found: v1/core". Default to
    # the same directory _resolve_targets() assumes.
    plugin["workdir"] = step.working_dir or DEFAULT_WORKING_DIR
    # Every step gets the GPUs, including the ones upstream marks CPU-only.
    # Those markings mean "performs no GPU computation", not "runs without a
    # visible device": importing vllm loads _C_stable_libtorch.abi3.so, which
    # needs libcuda.so.1, so in a GPU-less container the import fails and the
    # tests collapse into "Failed to infer device type" and pydantic
    # ValidationErrors on ModelConfig. Build #15's V1 Others (CPU) came back
    # 194 failed / 208 passed for exactly this reason; the same three tests
    # pass with the devices attached and fail without them.
    #
    # Upstream is unaffected because its CPU steps run on a CPU-built image. We
    # have one CUDA image for everything. CPU steps still take a cpu_slots
    # concurrency slot rather than the single GPU one - see _concurrency() -
    # since they only need the device present, not idle.
    #
    # `gpus` is configurable because a shared box may have GPUs other people are
    # using: "all" would seize every device on the host. Pin it to the devices
    # this agent owns, e.g. '"device=0,1"'.
    plugin["gpus"] = _gpus_for(step, config)

    env = dict(config.get("env") or {})
    env.update(step.env)

    emitted: dict[str, Any] = {
        "label": step.label,
        "key": step.slug,
        "agents": {"queue": hardware.get("queue_gpu")},
        "plugins": [{"docker#v5.2.0": plugin}],
        "commands": _wrap_commands(step, config),
    }
    priority = _priority(step, config)
    if priority:
        emitted["priority"] = priority
    if _runs_pytest(step):
        emitted["artifact_paths"] = report.get("artifact_paths")
    if timeout:
        emitted["timeout_in_minutes"] = timeout
    if env:
        emitted["env"] = env
    # Give up the whole run once anything has failed. GPU steps here are
    # serialized on one box, so a queued step behind a failure is holding the
    # only GPU slot for a verdict nobody is waiting on - Sequence Parallel alone
    # takes 41 minutes. Cheap to re-run after a fix, expensive to sit through.
    # Set fail_fast: false to collect every failure in one pass instead.
    if config.get("fail_fast", True):
        emitted["cancel_on_build_failing"] = True

    # Serialize access to the GPUs. Buildkite runs steps in parallel by default,
    # and upstream can allow that because it routes each step shape to its own
    # elastic queue (gpu_1_queue / gpu_4_queue), so concurrent steps land on
    # different machines. Here every step targets one box with a fixed number of
    # GPUs and asks for `gpus: all`, so two concurrent steps would fight over the
    # same VRAM and fail in ways that look like flaky tests.
    #
    # concurrency_group is org-wide, so the name is scoped by queue: two agents on
    # different hardware must not block each other. It also holds across builds,
    # which is what we want - two builds sharing one box is the same collision.
    group, limit = _concurrency(step, config)
    if group:
        emitted["concurrency_group"] = group
        emitted["concurrency"] = limit
        # Duration varies hugely between steps, so let any ready job take the
        # slot rather than forcing creation order.
        emitted["concurrency_method"] = "eager"
    # Upstream shards some steps across parallel agents; a single local agent
    # gains nothing from that, and BUILDKITE_PARALLEL_JOB would be unset.
    return {k: v for k, v in emitted.items() if v is not None}


def build_pipeline(verdicts: list[Verdict], config: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}

    for verdict in verdicts:
        if verdict.mode is None:
            continue
        step = verdict.step
        entries = groups.setdefault(step.group or "ungrouped", [])
        if verdict.mode == MANUAL:
            entries.append(
                {
                    "block": f"Run {step.label}",
                    "key": f"block-{step.slug}",
                    "prompt": verdict.reason,
                }
            )
            emitted = emit_step(step, verdict.mode, config)
            emitted["depends_on"] = f"block-{step.slug}"
            entries.append(emitted)
        else:
            entries.append(emit_step(step, verdict.mode, config))

    # No selection-summary step here: nothing in the pipeline depends on it and
    # the test agent has no use for it. The report is the generating agent's own
    # output, written to a file it can upload as an artifact.
    steps: list[dict[str, Any]] = []
    for group in sorted(groups):
        steps.append({"group": group, "steps": groups[group]})
    return {"steps": steps}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def format_summary(
    verdicts: list[Verdict], config: dict[str, Any], diff_desc: str, run_all: str | None
) -> list[str]:
    hardware = config.get("hardware") or {}
    auto = [v for v in verdicts if v.mode == AUTO]
    manual = [v for v in verdicts if v.mode == MANUAL]
    dropped = [v for v in verdicts if v.mode is None]

    lines = [
        # First line is the headline, so a caller can `head -1` it for a log
        # section header without re-running the (file-walking) classification.
        format_headline(verdicts, config),
        "=" * 72,
        f"agent queue      : {hardware.get('queue_gpu')}",
        "gpu_count        : {}   arch: {}".format(
            hardware.get("gpu_count"), hardware.get("arch")
        ),
        f"image            : {config.get('image')}{_preset_suffix(config)}",
        f"diff             : {diff_desc}",
        f"run-all          : {run_all or 'no'}",
        "",
    ]

    selected = auto + manual
    total = len(verdicts) or 1

    def tally(label: str, n: int, note: str, denom: int = total) -> str:
        return f"{label:23s}: {n:3d}  {n / (denom or 1):6.1%}   {note}"

    lines += [
        f"STEPS   total upstream : {len(verdicts)}",
        tally(
            "        in pipeline",
            len(selected),
            f"(auto {len(auto)} + block {len(manual)})",
        ),
        tally("          auto", len(auto), "run on every build"),
        tally("          block", len(manual), "need a click in the Buildkite UI"),
        tally("        excluded", len(dropped), "this hardware cannot run them"),
    ]

    # Group-level view: a group is "covered" when at least one of its steps runs
    # without a click. Covering a group partially still exercises the area.
    by_group: dict[str, list[Verdict]] = {}
    for v in verdicts:
        by_group.setdefault(v.step.group or "ungrouped", []).append(v)
    groups_total = len(by_group)
    g_auto = {g for g, vs in by_group.items() if any(x.mode == AUTO for x in vs)}
    g_block = {
        g
        for g, vs in by_group.items()
        if g not in g_auto and any(x.mode == MANUAL for x in vs)
    }
    g_none = set(by_group) - g_auto - g_block

    lines += [
        "",
        f"GROUPS  total upstream : {groups_total}",
        tally(
            "        auto-covered",
            len(g_auto),
            ">=1 step runs unattended",
            groups_total,
        ),
        tally(
            "        block-only",
            len(g_block),
            "reachable, but only by hand",
            groups_total,
        ),
        tally(
            "        no coverage", len(g_none), "nothing runnable here", groups_total
        ),
        "",
        "--- GROUP COVERAGE ---",
        f"  {'group':28s} {'auto':>4s} {'blk':>4s} {'excl':>4s}  coverage",
    ]
    for group in sorted(by_group):
        vs = by_group[group]
        n_a = sum(1 for x in vs if x.mode == AUTO)
        n_b = sum(1 for x in vs if x.mode == MANUAL)
        n_x = sum(1 for x in vs if x.mode is None)
        if n_a:
            mark = "auto" if not n_x else f"auto ({n_a}/{len(vs)} steps)"
        elif n_b:
            mark = "block only"
        else:
            mark = "none"
        lines.append(f"  {group[:28]:28s} {n_a:4d} {n_b:4d} {n_x:4d}  {mark}")

    lines += ["", f"--- AUTO ({len(auto)}) ---"]

    def describe(v: Verdict) -> str:
        tag = "cpu" if _is_cpu_only(v.step) else f"{v.step.num_devices}gpu"
        tier = v.step.cost.tier if v.step.cost else "--"
        return (
            f"  [{v.step.group[:18]:18s}] {tag:>4s} {tier:>2s}  "
            f"{v.step.label[:40]:40s} {v.reason}"
        )

    for v in sorted(auto, key=lambda x: (x.step.group, x.step.label)):
        lines.append(describe(v))

    lines += ["", f"--- MANUAL ({len(manual)}) ---"]
    for v in sorted(manual, key=lambda x: (x.step.group, x.step.label)):
        lines.append(describe(v))

    lines += ["", "--- COST PROFILE (selected steps) ---"]
    header = (
        f"  {'tier':4s} {'step':40s} {'files':>5s} {'srv':>4s} {'eng':>4s} "
        f"{'mdl':>4s} {'cmpl':>4s} {'tp>1':>4s} {'upstream':>8s}"
    )
    lines.append(header)
    for v in sorted(
        auto + manual,
        key=lambda x: (
            TIER_ORDER.index(x.step.cost.tier)
            if x.step.cost and x.step.cost.tier in TIER_ORDER
            else len(TIER_ORDER),
            x.step.label,
        ),
    ):
        c = v.step.cost
        if not c:
            continue
        lines.append(
            f"  {c.tier:4s} {v.step.label[:40]:40s} {c.files:5d} {c.server_files:4d} "
            f"{c.engine_files:4d} {c.models:4d} {c.compile_files:4d} "
            f"{c.multigpu_files:4d} {str(v.step.timeout_in_minutes or '-'):>7s}m"
        )
    return lines


def format_headline(verdicts: list[Verdict], config: dict[str, Any]) -> str:
    """One scannable line, for a collapsed log section header."""
    hardware = config.get("hardware") or {}
    auto = sum(1 for v in verdicts if v.mode == AUTO)
    block = sum(1 for v in verdicts if v.mode == MANUAL)
    groups = {v.step.group or "ungrouped" for v in verdicts}
    covered = {v.step.group or "ungrouped" for v in verdicts if v.mode == AUTO}
    name = hardware.get("name") or hardware.get("queue_gpu") or "pipeline"
    return (
        f"{name}: {auto} auto + {block} block of {len(verdicts)} steps, "
        f"{len(covered)}/{len(groups)} groups auto-covered"
    )


def format_excluded(verdicts: list[Verdict]) -> list[str]:
    dropped = [v for v in verdicts if v.mode is None]
    reasons: dict[str, int] = {}
    for v in dropped:
        key = re.sub(r"\d+", "N", v.reason)
        reasons[key] = reasons.get(key, 0) + 1
    lines = [f"--- EXCLUDED ({len(dropped)}) by reason ---"]
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        lines.append(f"  {count:4d}  {reason}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--test-areas", type=Path, default=DEFAULT_TEST_AREAS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the selection to stderr instead of emitting a pipeline",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="ignore source_file_dependencies and select every viable step",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="also write the selection report here, for upload as an artifact",
    )
    parser.add_argument(
        "--headline",
        action="store_true",
        help="print a one-line summary to stdout and exit; for a log section header",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    steps = load_steps(args.test_areas)
    files, diff_desc = changed_files()

    forced = matches_run_all(files, config)
    run_all = args.run_all or bool(forced) or not files
    run_all_desc = None
    if args.run_all:
        run_all_desc = "--run-all"
    elif forced:
        run_all_desc = f"changed {forced}"
    elif not files:
        run_all_desc = "empty diff"

    verdicts = [classify(s, config, files, run_all) for s in steps]

    if args.headline:
        print(format_headline(verdicts, config))
        return 0

    report = format_summary(verdicts, config, diff_desc, run_all_desc) + [""]
    report += format_excluded(verdicts)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(report) + "\n")

    if args.dry_run:
        print("\n".join(report))
        return 0

    # stdout carries only the pipeline, so it can be piped straight to
    # `buildkite-agent pipeline upload`. The report goes to stderr so it still
    # reaches the step log when no --report file was asked for; with --report the
    # caller displays the file instead, and echoing here would double it.
    if not args.report:
        print("\n".join(report), file=sys.stderr)
    pipeline = build_pipeline(verdicts, config)
    yaml.safe_dump(pipeline, sys.stdout, sort_keys=False, default_flow_style=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
