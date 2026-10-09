# 1233_rd73_stable_graph_cache_key: stable FNV-1a shape-fingerprint key for the HIP/CUDA graph cache (RD73)

Patch id: `1233_rd73_stable_graph_cache_key`. Plan item: `RD73`. Bound
Experiment Contract: `RD73-STABLE-GRAPH-CACHE-KEY`
(`config/experiment-contracts.toml`).

## Scope

Target architecture: gfx1100 (contract `scope.architectures`). Backend:
HIP. Ported byte-for-byte from mrlordcat-rdna-lab commit
`7f2e7e4a3ebf8e3b5aade75743c267f5ad7df199`.

Replaces `ggml_cuda_graph_get_key()`'s use of the raw first-node
pointer as the `cuda_graphs` map key with a 64-bit FNV-1a fingerprint
over node count plus the first/last nodes' op/name/`ne[]`, which is
stable across allocations for a recurring shape. The existing per-node
`memcmp` correctness check in `ggml_cuda_graph_update_required()` is
unchanged -- a fingerprint collision only costs an extra recapture,
never stale-graph reuse.

## Historical evidence is not current

Upstream-fork's own measured data (3.8k-node Qwen3.6-27B graph,
repeated speculative-verify batches): verify ubatch sync 150ms ->
57ms. **This has not been reproduced on BigCherry hardware.**

Production dual-XTX/27B instrumentation (2026-08-23) confirmed the
underlying mechanism directly: recapture rate control=49.6% vs
treatment=38.9% (~22% relative reduction) -- the fingerprint key really
does reduce cold-cache misses on recurring verify shapes. An earlier,
sequential-with-debug-logging production throughput benchmark found a
mixed result (tg128 -15.6%, tg2048 +16.6%); a **later, definitive clean**
(interleaved control/treatment, no debug logging) 10-round production
benchmark **superseded it**: all 9 configs' throughput deltas fell
within 0.1-2.3% of control -- the earlier mixed result was a
measurement artifact of the sequential-with-debug-logging methodology,
not a real effect. RD73 is throughput-neutral at production scale
under clean measurement; the mechanism is real, but no positive
production-scale throughput win is currently established (this
contract's `end_to_end_gain_pct = 3.0` requires a fresh, real MTP
verify-workload result to satisfy, deliberately set above the observed
noise floor).

VA06's own bounded resource-characterization run (2026-09-01, 3 reps
per arm, fully deterministic): peak `graph_cache_entries` control=386,
subject=651 (~68.7% relative increase) under a fixed repeated-shape MTP
completion burst -- the mechanism trades increased cache-key
cardinality (more distinct shapes retained as separate entries, since a
stable fingerprint no longer collapses non-identical shapes onto a
shared allocation-dependent pointer) for higher hit rates. This
contract's `resource_limits.graph_cache_entries.max_value = 800` is
subject peak (651) plus ~23% documented engineering headroom. Full
methodology is recorded in `docs/planning/active/validation-package-standard/VA06.md`.

## How to invoke validation

Real, authoritative full-qualification path (VA06). PA36 RD73 legacy
compatibility retirement: RD73 no longer has a dedicated
`--run-rd73-contract` CLI flag. It runs exclusively through the generic
patch-local producer (`validation/producer.toml`'s `[producer.rd73]`,
entrypoint `producer.py:run`) via `--validation-producer`:

```
PYTHONPATH=tools python -m bigcherry.patch.validation_campaign \
  --patch 1233_rd73_stable_graph_cache_key \
  --validation-producer 1233_rd73_stable_graph_cache_key/rd73 \
  --model <tierL-qwen27b-q8.gguf> \
  --producer-corpus tools/bigcherry/bench/corpora/mtp-27b-v1.jsonl \
  --device-map gfx1100=0,1 \
  --hip-path <production-rocm> --amdgpu-targets gfx1100 \
  --workdir <fresh-workdir>
```

Two preflight prerequisites (both fail closed before any hardware use):
the producer's `require_device_visibility(exact_count=2, env=ctx.build_env)`
reads the **ambient** `HIP_VISIBLE_DEVICES` (from `ctx.build_env`, which
copies `os.environ`), so the invoking shell must expose **exactly two**
distinct devices (e.g. `export HIP_VISIBLE_DEVICES=0,1`); `--device-map
gfx1100=0,1` is the per-architecture device-index mapping (`ARCH=ID[,ID...]`)
and does not itself satisfy that check. RD73 also passes `ctx.model` into
every `AttestedServerSession`, so `--model` is mandatory (the
`--validation-producer` path does not re-impose the legacy parser's
model-required check). The producer now fails fast on a missing model

`producer.py:run()` executes RD73's real paired MTP-verify performance
lane over a real llama-server HTTP harness (`_run_mtp_server_lane`),
activation evidence read from that SAME lane's own control/subject
server log files (no second server/model load), a dedicated subject-only
graph-cache resource burst session (`_run_resource_burst`), decode control
lane over a second real llama-server pair driven via the documented Brutus
bench runner (`_run_decode_control_lane` + `run_bench_runner_server_bench()`,
`docs/reference/testing/TEST.md`'s "Server benchmark (Brutus bench
runner)" section), and bit-identical correctness
(`_check_bit_identical`, reusing the MTP lane's own retained
request/response pairs), then composes them into a real
`evaluate_promotion_gate()` verdict (PASS/FAIL/INVALID). Real
llama-bench is never used anywhere in this path -- it proved unworkable
for RD73's real 27B/dual-GPU/-sm-tensor config on real Brutus hardware
(repeated crashes: OOM under resource contention with production traffic,
and a hard argument-parse error for `--fit`, which llama-bench does not
even register).

**Real hardware constraint (VA06):** control and subject llama-server
processes can never run concurrently for this 27B model -- each needs
~13GB/GPU under `-sm tensor` split, and two copies exceed the 24.5GB/GPU
Brutus dual-XTX cards (a real `cudaMalloc` out-of-memory abort, confirmed
on hardware). The MTP performance lane and decode control lane both
launch one fresh server per single measured/warmup request, alternating
control/subject arms sequentially -- this preserves the alternating-order
discipline this project's own prior production benchmarking found
necessary (see "Historical evidence is not current" above), at the cost
of a full server/model reload per request. Because a fresh process
resets the in-memory graph cache every restart, the resource lane
cannot reuse these same servers -- `_run_resource_burst()`
launches one long-lived subject-only server (no concurrent control, so
no VRAM conflict) and drives a real repeated-shape request burst against
it, matching this contract's own documented characterization
methodology.

**Evidence binding (VA06):** the producer's own contract-level
PASS/FAIL/INVALID verdict is real and auditable -- it is written into the
run's `producer-execution.json` / persisted validation evidence (the
producer emits exactly the eight `producer.toml`-declared artifacts; there
is no separate `rd73-contract-qualification.json`). With
`standard_campaign = "run"`, `_bind_producer_result_evidence()` binds the
producer's performance, correctness, trace, and activation evidence into
the validation context **before** `evaluate_check()`, so the generic
`validation.toml` checks (benchmark, correctness-summary, trace-marker)
consume real bound values -- they are **not** BLOCKED.
`eligible_for_validated_state` additionally requires both the adapter
verdict and the bound contract promotion to pass; the tracked status stays
`rejected` regardless (see "Known limitations").

## Known limitations

- **Performance/controls have a real producer (VA06).** RD73's own
  `_run_mtp_server_lane()` (paired control/subject llama-server
  processes, real MTP-verify HTTP requests, client-measured `wall_tps`)
  and `_run_decode_control_lane()` (a second paired real
  llama-server pair, driven via the documented Brutus bench runner --
  never llama-bench, which is unworkable for this real 27B/dual-GPU
  config) produce real evidence, composed via `aggregate_contract_effects()`
  against the contract's own `end_to_end_gain_pct`/
  `max_control_regression_pct` thresholds. With `standard_campaign =
  "run"`, the generic `performance`/`controls` checks (validator="benchmark")
  consume that bound producer performance evidence via
  `_bind_producer_result_evidence()` (not BLOCKED).
- **Correctness (`bit_identical`) has a real producer (VA06).**
  `_check_bit_identical()` performs exact string-equality
  comparison of the MTP lane's paired control/subject generated content,
  failing closed on mismatch/missing/non-string/unpaired records.
  With `standard_campaign = "run"`, the generic `correctness` check
  (validator="correctness-summary") consumes that bound producer
  correctness evidence via `_bind_producer_result_evidence()` (not
  BLOCKED).
- **Activation has a real marker probe (VA06).** RD73's patch source
  now carries a `BIGCHERRY_PATCH_TRACE`-gated marker at the stable-key
  execution site (`BIGCHERRY_PATCH_HIT patch=1233_rd73
  path=stable_graph_cache_key`), and the producer produces a real
  subject-hit/control-miss result by reading that marker from the MTP
  lane's own control/subject server log files; with `standard_campaign =
  "run"`, the generic `activation` check (validator="trace-marker")
  consumes that bound producer trace evidence via
  `_bind_producer_result_evidence()` (not BLOCKED).
  The generic tune-binary/`GGML_CUDA_DISABLE_FUSION`-based negative
  control is still **not** valid for this patch (RD73 is graph-cache
  keying, not a fusion path `GGML_CUDA_DISABLE_FUSION` controls) and
  must never be reused here -- the RD73-specific probe above is the
  correct control instead.
- **`resource_limits` has a real producer (VA06).** The contract's
  `graph_cache_entries` bound is real (derived from VA06's actual
  measurement). The producer now has real, tested
  `_parse_resource_telemetry()` (fails closed on any malformed
  `BIGCHERRY_RD73_RESOURCE`-prefixed line) and a peak-of-subject-readings
  reduction (-> `ResourceResult`, no paired control required) inlined into
  `run()`, driven by
  the patch's `BIGCHERRY_RD73_RESOURCE_TRACE`-gated telemetry.
- **Tracked status is governance-gated, not evidence-gated.** The
  adapter checks and the bound contract promotion are real, but this
  patch's tracked-status stays `rejected` (its `patch.toml`
  `state = "rejected"`, a deliberate governance decision after HI162's
  better-controlled A/B measured regression). A real hardware
  `--validation-producer 1233_rd73_stable_graph_cache_key/rd73`
  qualification alone does not update tracked status; the governance
  decision must be reversed first, and any revision requires fresh
  accepted evidence.

## Control vs. subject

Standard validation-domain composition: `control_src` (this patch
absent -- graph-cache key remains the raw first-node pointer) vs.
`patched_src`/validation-subject (this patch present -- key is the
stable FNV-1a fingerprint). No patch-specific composition wrinkle.

## Evidence

Runtime artifacts (build logs, raw benchmark output) will land under
`artifacts/patch-validation/1233_rd73_stable_graph_cache_key/<campaign-identity>/`,
outside this tracked patch directory, once a real executor exists. The
compact, tracked record is `evidence/validation.json` (not yet
present -- no real validation run has produced one).
