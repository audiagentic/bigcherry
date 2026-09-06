# End-to-end tuning: `bigcherry tune-campaign`

A single-command orchestrator (HI130) for the full
record → tune → correctness-evidence → tuning-promotion → replay pipeline, driven
against real hardware. Before this existed, running the full pipeline meant
manually sequencing several separate `bigcherry build`/tuning invocations
and hand-carrying artifacts (inventory, measurements, promoted winners)
between stages — this collapses that into one command with one real
reproducible receipt.

## Basic usage

```bash
PYTHONPATH=tools python -m bigcherry tune-campaign \
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
4. **Tuning promotion** — writes the promoted-winners JSONL from whatever
   passed correctness evidence. This promotes tuning winners inside this
   campaign only; it is not patch, contract, plan, release, or production
   policy acceptance.
5. **Replay builds** — builds the production `replay` lane and the validation-only
   `replay-diagnostic` companion. Production excludes diagnostics. The companion
   enables dispatch coverage and replay-hit diagnostics. Before exporting,
   require matching source composition, recomputed catalog descriptors,
   generated registry/compile inputs, and all non-diagnostic requested CMake
   options. Requested-option parity is not observed compiler-option parity.
   The runtime bundle must carry `generated_inputs_verification=compiled-copy-v1`
   and the matching recomputed input digest: the worker verifies the actual
   `build_dir/generated-inputs` copy before configure, before compile and after
   compile. Historical builds without this proof cannot be attested retroactively.
6. **Replay export + verify** — exports the replay cache **against the
   production replay build's own manifest** (not the tune build's), then runs
   behavioral/coverage/recovery validation on the matched diagnostic companion.
   Receipt schema 4 retains `replay` as the production artifact and adds
   `replay_validation` for the observer; coverage records both build-plan IDs
   and `observation_role=diagnostic-companion`. This is not same-cell production
   performance activation proof. This ordering is load-bearing: an earlier version of this
   workflow exported against the tune stage's manifest, which produced a
   manifest-hash mismatch and silently invalidated every cache entry the
   moment a real replay server started — fixed by reordering to
   build → export → verify, always against the stage that will actually
   consume the cache.

A failure at any stage raises `TuneCampaignError` rather than continuing
past a stale or invalid state (e.g. `rerun_required` coverage at the verify
stage is a hard failure, not a warning).

## What this is not

`tune-campaign` finds and promotes tuned candidates within its own replay
workflow. That promotion is not a patch or release decision. It does not explain
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
