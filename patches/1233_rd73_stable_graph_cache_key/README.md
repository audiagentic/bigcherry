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

**No real validation producer is wired yet.** `validation.toml`
declares all 6 checks honestly against real validator shapes, but none
of correctness/activation/performance/controls has a real evidence
producer behind it in `validation_campaign.py` -- unlike RD04/RD08/RD58,
there is no `--run-rd73-...` CLI flag yet. Running the generic
adapter today reports every non-apply/build check as `BLOCKED`, not a
fabricated pass. Building the real executor (paired MTP-verify
performance/controls lanes, a real bit-identical correctness producer,
and a real subject-hit/control-miss activation marker) is separate,
future work -- see the "Known limitations" section below.

## Known limitations

- **Performance/controls have no real producer yet.** `validation.toml`
  declares `performance`/`controls` against the real `benchmark`
  validator shape, but no evidence is bound -- both report `BLOCKED`.
  A real executor needs to run the actual MTP verify workload
  (`--spec-type draft-mtp`) this contract's `positive.workloads =
  ["mtp_verify"]` names, paired against a `decode` control lane, to
  produce a real `end_to_end_gain_pct`/`max_control_regression_pct`
  result.
- **Correctness (`bit_identical`) has no real producer yet.**
  `validation.toml` declares the check honestly against a real
  validator shape (`backend-ops`, op label `RD73_MTP_BIT_IDENTICAL`),
  but no evidence is bound -- it reports `BLOCKED`, not a fabricated
  pass.
- **Activation has no real marker probe yet.** RD73's patch source
  carries no `BIGCHERRY_PATCH_TRACE`-gated marker. The generic
  tune-binary/`GGML_CUDA_DISABLE_FUSION`-based negative control is
  **not** valid for this patch (RD73 is graph-cache keying, not a
  fusion path `GGML_CUDA_DISABLE_FUSION` controls) and must never be
  reused here. `validation.toml` declares the check against the real,
  future exact marker text this patch would need to emit
  (`BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key`)
  -- stays declared but unsatisfied (`BLOCKED`) until a real
  subject-hit/control-miss probe exists.
- **`resource_limits` also has no real producer yet.** The contract's
  `graph_cache_entries` bound is real (derived from VA06's actual
  measurement), but nothing in `validation_campaign.py` yet produces a
  real `ResourceResult` to check against it during a validation run --
  VA06's characterization was a one-off, uncommitted, temporary-
  instrumentation experiment, not a repeatable evidence producer.
- These gaps mean this patch's tracked-status stays `untested` until a
  real RD73 executor exists; it cannot honestly claim `ported-benched`
  or `ported-validated` on the strength of the contract alone.

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
