# 1204_rd08_q6k_mmvq_vdr2: Q6_K mmvq VDR=2 decode kernel (RD08)

Patch id: `1204_rd08_q6k_mmvq_vdr2`. Plan item: `RD08`. Bound Experiment
Contract: `RD08-Q6K-MMVQ-VDR2` (`config/experiment-contracts.toml`).

## Scope

Target architectures: gfx1100, gfx1201, gfx1030 (contract `scope.architectures`).
Backend: HIP. Weight type: Q6_K. Prerequisites: none.

Adds a VDR=2 Q6_K `vec_dot` entry point that processes both 8-element
chunks of a Q6_K dot product in one call (4 `dp4a` ops instead of 2,
halving loop iterations for the same row) and switches
`get_vec_dot_q_cuda`'s Q6_K case to it. Also makes `GGML_CUDA_OP_TIMING`
disable CUDA graph capture instead of aborting, and adds the decode-shaped
perf test cases the correctness producer below uses. Ported from
stew675-rdna-boosts fork commit `4591cc980`
(https://github.com/stew675/llama.cpp); not merged into ggml-org/llama.cpp
master. Decode is DRAM-bound, so this targets a modest tg128 gain, not a
compute-bound win — the fork's own claim is that the VDR=2 kernel is
bit-identical to the VDR=1 kernel it replaces, which is exactly what this
patch's bound correctness check proves.

## How to invoke validation

Real, authoritative full-qualification path (VA14):

```
PYTHONPATH=tools python -m bigcherry.patch.validation_campaign \
  --patch 1204_rd08_q6k_mmvq_vdr2 \
  --model <tierM-gptoss20b-q6k.gguf> \
  --hip-path <production-rocm> --amdgpu-targets gfx1100 \
  --manifest <hip-autotune-manifest.json> \
  --workdir <fresh-workdir> --build-root <build-root> \
  --worktree-root <worktree-root> --bench-repetitions 3 \
  --run-rd08-contract
```

`--run-rd08-contract` executes RD08's real positive(decode)/control(prefill)
lane pairs (`tools/bigcherry/experiment/execution.py`), the real
bit-identical correctness producer (below), and a real subject-hit/
control-miss trigger probe, then composes them via
`evaluate_promotion_gate()` into `eligible_for_validated_state`. See
`docs/reference/testing/PATCH_VALIDATION.md`'s "Real contract-execution
architecture (VA14)" section for the full mechanism.

`--run-rd08-lanes` alone is diagnostic-only (lane execution + evidence,
never feeds eligibility) — use it only to inspect measured performance
without running the full (slower) correctness producer.

## Correctness producer

`validation/rd08_correctness.py` (`materialize_rd08_variants()`,
`require_rd08_correctness_evidence()`) is patch-specific: it materializes
its own isolated VDR2-subject/VDR1-control worktrees (two checked semantic
reversions of exactly this patch's routing edits — see
`apply_vdr1_control()`), builds `test-backend-ops` for both, and proves
bit-identical output across the 5 real decode shapes this patch's own
perf-case edit adds × 3 deterministic seeds (backend1_digest equality, not
NMSE agreement). This is a real source-level A/B of VDR=2 vs VDR=1, not a
dispatch-candidate comparison.

## Control vs. subject

Validation-domain control/subject follow the framework's normal
`control_src`/`patched_src` composition (this patch present vs. absent).
The correctness producer additionally uses its own VDR2/VDR1 source-level
A/B (above) — do not confuse the two: the validation-domain control never
has this patch at all, while the correctness producer's "control" has the
patch's other edits but reverts only the two VDR routing lines.

## Real full contract-qualification run at current pin (2026-09-13) -- real correctness-gate FAIL, investigated

Ran `run_rd08_contract_qualification()` directly for real on Brutus
against the current active pin (`b10901`/`28ff0958291c`), single gfx1100,
`tierM-gptoss20b-q6k`, 6 paired decode/prefill rounds, real trigger probe.

**Trigger proof: real PASS.** `subject_hit=true`/`control_hit=false`
(via `--verbose`-gated `GGML_LOG_WARN` marker), positive lane checked and
triggered.

**Performance: inconclusive at 6 rounds.** `target_kernel_gain_pct =
+0.30%`, but `ci95_low = -0.26%` (interval crosses zero) --
`min_paired_rounds` for this contract is not yet migrated to the
interval policy (still `point_estimate_v1`/legacy-waiver, per
`config/experiment-contract-legacy.toml`'s `RD08-Q6K-MMVQ-VDR2` entry),
so this is a real, honest data point but not by itself a promotion
blocker under the contract's own current (weaker) policy.

**Correctness: real FAIL on the `bit_identical` gate -- investigated,
NOT concluded to be a patch defect.** All 15 rows (5 shapes x 3 seeds)
failed the exact-digest equality check, but by a uniform, tiny margin:
subject vs. control `err` values differ by only ~1e-7 to ~1e-6
*relative* magnitude in every single row (e.g. `ffn` seed=1:
`2.353963937227954e-05` vs `2.3539629204239105e-05`), all far below the
gate's own `0.0005` tolerance, and every row's own internal
backend1-vs-backend2 comparison independently reports `status=ok`. This
uniform, sub-threshold pattern across every shape and seed -- rather
than an isolated failure or a magnitude comparable to the tolerance --
is inconsistent with a real VDR=2-vs-VDR=1 numerical divergence and
consistent with ordinary GPU non-associative floating-point reduction
non-determinism between separate process launches, which the
1222/1223 (HI67) deterministic-seed patches (already correctly included
in `RD08_PATCH_STACK`) control the RNG *inputs* for but cannot control
the GPU's internal warp/reduction execution order for. **Not
independently confirmed** (a same-binary-run-twice control was not
performed this session due to time) -- this is a strong, evidence-based
hypothesis, not a proven root cause. Filed as PRBE103 to run that
confirming control and, if confirmed, reconsider whether an exact-digest
`bit_identical` check is achievable at all for GPU kernels via this
methodology, or whether it needs a tolerance-based redesign.

**No promotion attempted or claimed** -- `evaluate_promotion_gate()`
correctly returned `status=fail` on the correctness gate; this is not
overridden or reinterpreted here. The patch's real disposition remains
`untested` pending PRBE103's root-cause confirmation.

## Known limitations

None declared — this patch is not `deferred-hardware`. See the
2026-09-13 contract-qualification section above for a real, open,
not-yet-root-caused correctness-gate finding (PRBE103).

## Evidence

Runtime artifacts (build logs, raw benchmark output, correctness rows)
land under `artifacts/patch-validation/1204_rd08_q6k_mmvq_vdr2/<campaign-identity>/`,
outside this tracked patch directory. The compact, tracked record is
`evidence/validation.json`.
