# Generated pipeline samples

Reference output from `.buildkite/local-scripts/gen_pipeline.py`. These are
**samples for review, not inputs** — nothing reads them at build time. The
generating agent produces the real pipeline fresh on every build and pipes it
straight to `buildkite-agent pipeline upload`.

| file | tracked | what it is |
| --- | --- | --- |
| `pipeline-4090.excerpt.yaml` | yes | Trimmed. One example of each step kind. Start here. |
| `pipeline-4090.sample.yaml` | no | The full uploaded pipeline: 15 groups, 26 test steps, 11 block steps. |
| `selection-4090.sample.txt` | no | The selection report: group coverage, rates, per-step cost profile. |

The two large files are gitignored — they are regenerable and go stale whenever
the config changes. Run the command below to produce them locally.

Regenerate:

```bash
python3 .buildkite/local-scripts/gen_pipeline.py --run-all \
  --report output/selection-4090.sample.txt \
  > output/pipeline-4090.sample.yaml
```

The two outputs go to different places on purpose. stdout is only ever the
pipeline, so it can be piped straight into `buildkite-agent pipeline upload`; the
report goes to `--report` because it is the *generating* agent's output. The test
agent never reads it, and no step depends on it, so it travels as a build
artifact instead of as a step in the pipeline.

Generated with `--run-all`, so the sample shows every step this hardware can
run. A real build selects a subset from the diff, and steps not selected become
block steps instead.

## What to look at

- **`agents.queue: gpu-4090`** on every test step — the generating agent runs on
  `local-mac` and only emits YAML; the GPU agent executes it.
- **`pid: private`** — several upstream PD scripts run an unfiltered
  `pkill -9 -f "vllm serve"`, which must not reach host processes.
- **`volumes`** — pre-staged model cache. Read-write on purpose: vLLM's weight
  loader takes a filelock inside the cache dir. `HF_HUB_OFFLINE=1` turns a
  missing model into a fast failure rather than a silent multi-GB download.
- **`gpus: all` absent on CPU-only steps** — 6 of 26 run in the container on the
  GPU host without reserving a GPU.
- **`BUILDKITE_PARALLEL_JOB` exports** — upstream shards some steps across
  parallel agents. We drop `parallelism`, so these default to the single-shard
  case (`0` of `1`); without them pytest gets an empty `--shard-id=` and fails
  at collection.
- **`concurrency_group` + `concurrency: 1`** on GPU steps — Buildkite runs steps
  in parallel by default, and every step here asks for `gpus: all` on the same
  box. Without this, concurrent steps contend for VRAM and fail like flaky tests.
  Upstream needs no equivalent: it routes each step shape to its own elastic
  queue, so its parallel steps land on separate machines. CPU-only steps get a
  separate group (`cpu_slots: 2`) since they do not touch VRAM.
- **`--junitxml` via `PYTEST_ADDOPTS`** — upstream command strings are never
  rewritten. `artifact_paths` uploads the report even when the step fails.
- **`timeout_in_minutes`** — upstream's value scaled by
  `filter.timeout_multiplier` (2.0), since a 4090 is slower than an H200.

The skip-ratio guard runs `--warn-only` until thresholds are calibrated on real
hardware: on SM89 every sm90+/sm100-gated test skips, and pytest exits 0 when
everything skips, so an all-skipped run would otherwise look green.
