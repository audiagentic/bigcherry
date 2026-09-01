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
  --model <tierA-qwen4b-q6k.gguf> \
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

## Known limitations

None declared — this patch is not `deferred-hardware`.

## Evidence

Runtime artifacts (build logs, raw benchmark output, correctness rows)
land under `artifacts/patch-validation/1204_rd08_q6k_mmvq_vdr2/<campaign-identity>/`,
outside this tracked patch directory. The compact, tracked record is
`evidence/validation.json`.
