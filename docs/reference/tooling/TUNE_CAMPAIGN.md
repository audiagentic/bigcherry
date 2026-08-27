# End-to-end tuning: `bigcherry tune-campaign`

A single-command orchestrator (HI130) for the full
record → tune → correctness-evidence → promote → replay pipeline, driven
against real hardware. Before this existed, running the full pipeline meant
manually sequencing several separate `bigcherry build`/tuning invocations
and hand-carrying artifacts (inventory, measurements, promoted winners)
between stages — this collapses that into one command with one real
reproducible receipt.

## Basic usage

```bash
PYTHONPATH=tools python3 -m bigcherry tune-campaign \
    --platform linux-multi \
    --model /path/to/model.gguf \
    --devices 0,1 \
    --runtime-profile production-dual-xtx \
    --run-id my-tune-01 \
    --json
```

- `--runtime-profile <name>` — a named `[runtime-profile.<name>]` bundle
  from `config/recipes.toml`: server args, a **tune-context** size
  (deliberately smaller than production — the tuner's per-candidate timing
  workspace needs VRAM headroom beyond weights+KV-cache) and a
  **production-context** size, plus a minimum free-VRAM-per-device
  headroom. `production-dual-xtx` matches the real production launch
  profile; `production-safe-single` is a conservative single-GPU fallback.
- `--tune-screen-samples` / `--tune-final-samples` — sample counts for the
  tuner's two-phase screen-then-confirm measurement strategy.
- `--correctness-seeds` — how many seeds the correctness-evidence stage
  measures against before a candidate is eligible for promotion.
- `--q` / `--threshold-pct` / `--resamples` — statistical parameters for
  the promotion decision (bootstrap resampling, significance threshold).
- `--workdir` — defaults to `work_root/tune-campaigns/<run_id>`; every
  stage's real artifacts (record inventory, tune measurements, correctness
  evidence, promoted-winners JSONL, replay build/verification output) land
  here.

## Stage sequence

1. **Record** — builds and runs the `record` lane against the real model,
   discovering which dispatch signatures the workload actually issues.
2. **Tune** — builds and runs the `tune` lane, measuring candidates for
   every recorded signature at production tolerances.
3. **Correctness evidence** — validates promotion-eligible candidates
   against the configured seed count before they're allowed to promote.
4. **Promote** — writes the promoted-winners JSONL from whatever passed
   correctness evidence.
5. **Replay build** — builds the `replay` lane, which applies the promoted
   winners without re-measuring.
6. **Replay export + verify** — exports the replay cache **against the
   replay build's own manifest** (not the tune build's), then verifies
   coverage. This ordering is load-bearing: an earlier version of this
   workflow exported against the tune stage's manifest, which produced a
   manifest-hash mismatch and silently invalidated every cache entry the
   moment a real replay server started — fixed by reordering to
   build → export → verify, always against the stage that will actually
   consume the cache.

A failure at any stage raises `TuneCampaignError` rather than continuing
past a stale or invalid state (e.g. `rerun_required` coverage at the verify
stage is a hard failure, not a warning).

## What this is not

`tune-campaign` finds and promotes tuned candidates. It does not explain
*why* the hardware spends time where it does — for real kernel-level
profiling (rocprofv3 kernel/timing/resource data), see
[PROFILING.md](PROFILING.md)'s `bigcherry profile-campaign`. The two are
complementary: profile first to understand the cost structure, tune once
you have evidence of where the real opportunity is.

## Measurement rigor

Benchmark comparisons drawn from tune-campaign output (or from any manual
A/B against its artifacts) inherit the project's general measurement
discipline — see [TEST.md](../testing/TEST.md) for the seeded/deterministic
sampling requirements. Two real harness bugs were found and fixed this way
in this project's history: unseeded sampling temperature and an unseeded
synthetic-prompt generator, both of which had been perturbing
speculative-decoding draft-acceptance rate and producing false "tuning
signal" — the fix was `temperature=0.0, seed=42` for completions and a
deterministic per-prompt seed for prompt generation. Any new benchmark
harness work should default to deterministic sampling from the start
rather than discover this the same way.
