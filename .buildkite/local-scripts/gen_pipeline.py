#!/usr/bin/env python3
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
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / ".buildkite" / "ci_config_4090.yaml"
DEFAULT_TEST_AREAS = REPO_ROOT / ".buildkite" / "test_areas"


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
        return yaml.safe_load(handle) or {}


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
    work = (step.working_dir or "/vllm-workspace/tests").replace("/vllm-workspace", "")
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


def _wrap_commands(step: Step, config: dict[str, Any]) -> list[str]:
    """Add JUnit reporting and the skip-ratio guard around upstream commands.

    Upstream commands are never rewritten: pytest gets its options through
    PYTEST_ADDOPTS so the report lands in a known place without touching any
    command string.
    """
    # Steps that never call pytest (static audits, cargo, shell checks) get no
    # report, so wrapping them would only add a spurious skip-ratio check.
    if not _runs_pytest(step):
        return list(step.commands)

    report = config.get("report") or {}
    junit_dir = report.get("junit_dir") or "/vllm-workspace/test-reports"
    junit = _junit_path(step, report)
    # Absolute so it resolves regardless of the step's working_dir.
    checker = report.get(
        "checker_path", "/vllm-workspace/.buildkite/local-scripts/check_skip_ratio.py"
    )

    out = [f"mkdir -p {junit_dir}"]
    # We drop upstream's `parallelism`, so Buildkite never sets the two shard
    # variables. Commands that pass them to pytest would expand to an empty
    # `--shard-id= --num-shards=` and fail at collection, so define them as the
    # single-shard case. Harmless for steps that do not shard.
    out.append("export BUILDKITE_PARALLEL_JOB=${BUILDKITE_PARALLEL_JOB:-0}")
    out.append("export BUILDKITE_PARALLEL_JOB_COUNT=${BUILDKITE_PARALLEL_JOB_COUNT:-1}")
    # PYTEST_ADDOPTS applies to every pytest process the step spawns, including
    # the ones launched from upstream's shell wrappers.
    out.append(f'export PYTEST_ADDOPTS="--junitxml={junit} -o junit_family=xunit2"')
    out.extend(step.commands)

    enforce = bool(report.get("skip_ratio_enforce", False))
    threshold = report.get("skip_ratio_default", 0.9)
    guard = f"python3 {checker} {junit} --max {threshold}"
    if not enforce:
        guard += " --warn-only"
    # `|| true` keeps a warn-only guard from masking the real exit status.
    out.append(guard if enforce else f"{guard} || true")
    return out


def _concurrency(step: Step, config: dict[str, Any]) -> tuple[str | None, int]:
    """Return (concurrency_group, limit) for a step, or (None, 0) to leave it free.

    GPU steps share one group so only `gpu_slots` run at a time. CPU-only steps
    get their own group: they do not touch VRAM, so they can run alongside GPU
    work, but they still consume host RAM and CPU on the same box.
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

    # One GPU step at a time by default. Raising this only makes sense if the
    # steps' num_devices sum to no more than the box's GPU count, which is not
    # something this generator can guarantee, so it stays a manual choice.
    limit = int(rules.get("gpu_slots") or 1)
    return f"{prefix}/{queue}/gpu", limit


def emit_step(step: Step, mode: str, config: dict[str, Any]) -> dict[str, Any]:
    hardware = config.get("hardware") or {}
    rules = config.get("filter") or {}
    report = config.get("report") or {}

    multiplier = float(rules.get("timeout_multiplier") or 1.0)
    timeout = step.timeout_in_minutes
    if timeout:
        timeout = max(1, int(round(timeout * multiplier)))

    docker = config.get("docker") or {}
    plugin: dict[str, Any] = {
        "image": config.get("image"),
        "always-pull": bool(docker.get("always_pull", False)),
        "propagate-environment": True,
        # Never share the host PID namespace: several upstream PD scripts run an
        # unfiltered `pkill -9 -f "vllm serve"`.
        "pid": "private",
        # vLLM's multiprocessing needs more shared memory than docker's 64MB.
        "shm-size": docker.get("shm_size", "8gb"),
    }
    if docker.get("volumes"):
        plugin["volumes"] = list(docker["volumes"])
    if step.working_dir:
        plugin["workdir"] = step.working_dir
    # CPU-only steps still run in the container on the GPU host (upstream's
    # commands assume Linux), but must not reserve the GPUs.
    if not _is_cpu_only(step):
        plugin["gpus"] = "all"

    env = dict(config.get("env") or {})
    env.update(step.env)

    emitted: dict[str, Any] = {
        "label": step.label,
        "key": step.slug,
        "agents": {"queue": hardware.get("queue_gpu")},
        "plugins": [{"docker#v5.2.0": plugin}],
        "commands": _wrap_commands(step, config),
    }
    if _runs_pytest(step):
        emitted["artifact_paths"] = report.get("artifact_paths")
    if timeout:
        emitted["timeout_in_minutes"] = timeout
    if env:
        emitted["env"] = env

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
        f"image            : {config.get('image')}",
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
