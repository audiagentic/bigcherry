# Deep profiling: `bigcherry profile-campaign`

A repeatable, rocprofv3-based deep-profiling command (HI132) for real
kernel-level GPU measurement: kernel names, call counts, timing
(mean/p95/total), and resource usage (VGPR/SGPR/scratch), plus real
HIP/HSA/RCCL API and memory-copy traces. Complements the tuning workflow
(`tune-campaign`, see [TUNE_CAMPAIGN.md](TUNE_CAMPAIGN.md)) — this is for
*understanding why* a workload spends time where it does, not for finding
or promoting tuned candidates.

There is no mock/simulation mode: this always drives a real `rocprofv3`
subprocess against a real running `llama-server`, on real hardware.

## What it does

For a given build lane, model, device set, and runtime profile, the
campaign:

1. Builds (or reuses a cached build of) the requested lane.
2. Runs several **unprofiled control blocks** (default 10 reps each) to
   establish the environment's own noise floor — reported as `tg mean t/s`
   / stddev per block, with an explicit `environment_stable` verdict
   (control-block mean spread vs. a 5% threshold). If the environment is
   unstable, the report says so explicitly and **no comparative performance
   conclusion is drawn** — the kernel/timing data captured is still real
   and usable for diagnostic purposes, just not for A/B comparison.
3. Runs the requested number of **profiled GPU passes** (default 2, to
   check pass-to-pass reproducibility, not statistical power), each
   wrapped in `rocprofv3 --sys-trace` (HIP + HSA + kernel dispatch +
   memory-copy + RCCL API coverage).
4. Parses the resulting CSVs into a `ProfileReport` (per-pass kernel
   tables, capture status, RCCL-activity detection) and renders both JSON
   and Markdown.

## Basic usage

```bash
PYTHONPATH=tools python3 -m bigcherry profile-campaign \
    --platform linux-multi \
    --model /path/to/model.gguf \
    --devices 0,1 \
    --runtime-profile production-dual-xtx \
    --workload my-workload-label \
    --run-id my-run-01 \
    --control-reps 10 \
    --profile-passes 2 \
    --json
```

- `--runtime-profile <name>` — a named `[runtime-profile.<name>]` bundle
  from `config/recipes.toml` (server args, tune/production context sizes,
  VRAM headroom). Same convention `tune-campaign` uses.
- `--devices` — `HIP_VISIBLE_DEVICES` value, e.g. `0,1` or `0,1,2,3`.
- `--experiment <name>` — build with a named `[experiment.<name>]` patch
  overlay (e.g. to profile an untested/experimental patch in isolation —
  see [PATCH_SYSTEM.md](../patches/PATCH_SYSTEM.md) for how experiments
  compose with a source's base patch set).
- `--workdir` — defaults to `work_root/profile-campaigns/<run_id>`. The
  report, receipt, and every per-pass rocprofv3 output directory (kernel
  trace, memory-copy trace, HIP/HSA/RCCL API traces, agent info) land here.

Environment variables read by the *profiled process itself* (not the
campaign tool) still apply as normal — e.g. `GGML_HIP_REDUCE_PLAN=meta` to
force the META reduction path, or `GGML_HIP_REDUCE_TELEMETRY=<path>` to
enable the reduction observability JSONL (see
[MULTI_GPU_DISPATCH.md](../architecture/MULTI_GPU_DISPATCH.md)). Export
them before invoking the campaign; they're inherited by the server
subprocess exactly as they would be for a manual `llama-server` launch.

## Reading the report

- **`capture status: complete`** requires every expected GPU agent to
  appear in the kernel trace. RCCL activity is required in addition **only
  when the run's expected reduction provider is `auto` or `rccl`** — a
  deliberately META-forced run is not penalized for showing no RCCL
  activity (fixed in HI134 after a real false-positive: the detector now
  requires an actual collective call like `ncclAllReduce`/`ncclGroupEnd`,
  not just any row in the RCCL trace — `ncclGetVersion`/`ncclCommInitAll`
  etc. appear even when RCCL never runs a real collective).
- **`environment stable: False`** with a control-block spread — treat the
  kernel data as real and diagnostically useful, but do not draw a
  before/after performance conclusion from that run alone. Re-run, or pair
  with a separate controlled A/B (see the HI35/RV19 "impact model" tooling
  for that — `bigcherry impact`, `bigcherry kernel-fraction`).
- **CPU profile: unavailable** — `perf` is not currently usable in this
  environment (kernel/package mismatch on Brutus). Tracked as HI133,
  deliberately deferred; do not attempt to route around it by changing
  `perf_event_paranoid` — that's not the actual blocker.
- **Per-kernel table** — calls, total/mean/p95 microseconds, and
  VGPR/SGPR/scratch resource usage, straight from rocprofv3's
  `*_kernel_trace.csv`. Useful for spotting register-pressure or
  scratch-memory regressions between two builds (this is exactly how RD33
  was diagnosed: a suspected performance regression turned out to be
  ambient noise, confirmed by comparing VGPR/SGPR/scratch between builds —
  identical, ruling out a register-pressure explanation).

## Attribution instrumentation (advanced)

For questions the kernel trace alone can't answer — e.g. "which logical
operation issued this specific copy" — there's an optional, narrowly-scoped
attribution facility in the META reduction path (`allreduce_fallback` in
`ggml/src/ggml-backend-meta.cpp`), added for HI134's investigation into an
otherwise-unexplained `copyBufferRectAligned` cost:

- Enable with `GGML_HIP_REDUCE_TELEMETRY=<path>` (writes one JSONL event
  per reduction) plus the `hi134-meta-stage-trace` experiment patch
  (`[experiment.hi134-meta-stage-trace]` in `config/recipes.toml`, requires
  `0830_split_reduce_telemetry`, which is already in every default build).
- Each JSONL event's `meta_trace.stages[]` records every `FOLD`/
  `BUTTERFLY`/`COPY_BACK` transfer META submits: phase, step, source/dest
  logical rank, byte count, and tensor shape/strides — a bounded,
  allocation-free, thread-local capture (32 stages max) with **no added
  synchronization**, so it doesn't perturb the timing it's measuring.
- This is intentionally a diagnostic-only, `STATE=untested` patch (patch
  1242) — not something to build into production. See HI134 in
  `docs/planning/` for how this was used to conclusively separate a real
  anomaly from META's actual reduction cost.

## Real example: this is how HI134 was investigated

1. Baseline profile-campaign runs across several device topologies
   (`{0,1,2}`, `{0,1,2,3}`, `{0,1,3}`) with `GGML_HIP_REDUCE_PLAN=meta`
   forced, to isolate a suspicious kernel (`copyBufferRectAligned`) by
   varying one topology factor at a time (device count, which specific
   device participates, split mode).
2. A `-sm layer` negative control (never touches `comm_ctx`/META at all)
   on the same device set, to test whether the anomaly was reduction-
   related at all.
3. Once narrowed to "needs tensor-split + a specific device", the
   attribution instrumentation above was added to get an exact count and
   byte-size breakdown of META's own real transfers in the same profiled
   window — which, combined with `rocprofv3`'s `memory_copy_trace.csv`
   (source/destination agent + direction + timestamps per copy), gave a
   decisive answer: the anomaly was an *intra-device* copy unrelated to
   META's *cross-device* reduction traffic. See HI134/HI135 in
   `docs/planning/` for the full trail.

This workflow — vary one real-hardware factor at a time, escalate to
purpose-built instrumentation only once cheaper controls have narrowed the
search space, and never invent a conclusion the data doesn't support — is
the intended way to use this tool for anything beyond a routine
before/after comparison.
