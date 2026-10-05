---
id: BCOP29
order: 29
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T11:09:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Execute semantic-first sparse expert range qualification

## Description

Action/disposition record for the 2026-10-05 MET02 deep audit. MET02 remains the technical authority. The audit found that planned patch 1281 is not implemented on the live branch, so downstream work must prove the range semantic primitive before attempting compact HIP dispatch.

## Actions

1. Implement/mock MET02 Phase A on the current pin: CPU/reference `ggml_mul_mat_id_range` semantics plus boundary/full-range/inactive-lane tests. Do not add GPU compaction yet.
2. Confirm current native `MUL_MAT_ID` precision and MMQ J/tail ownership (#29911 and #27792/#29941/#29953 lineage) and reuse it rather than copying formulas into 1281.
3. Build the CPU reference/tests; then add the correctness-only HIP range path at the existing expert-ID grouping ownership boundary and build gfx1100/gfx1201.
4. Record expected versus observed active rows and transfer/work accounting before accepting any timing result.
5. Only if correctness-only range execution is clean, prototype compact grouped `(output_lane, local_expert)` execution using existing grouping/workspace. Remove the prototype if MET02's rejection gate fires.
6. Hand any proven architecture/density crossover to HIP-autotune; do not leave a second dispatch table in 1281.

## Evidence / consolidation

- No `patches/1281_moe_mul_mat_id_range/` exists on the audited live branch; MET02 was design-only at audit time.
- vLLM MoonEP demonstrates grouped expert execution over already-contiguous expert segments without a second permute/unpermute; use only that representation principle, not CUDA/NVLink assumptions.
- vLLM DeepEP uses invalid/non-local expert sentinels; BigCherry retains MET02's stricter exact-zero inactive-lane contract.
- llama.cpp #29963 adds host-RAM pipeline parallelism through scheduler copies/events. It is a separate whole-layer mechanism; do not import that scheduler machinery into 1281. If #29963 removes the target need for tiered experts, reassess MET03 rather than expanding MET02.

## Completion / disposition

Close this item when MET02 reaches one of these states:

- **promoted:** Phase A/B correctness passes and compact dispatch meets MET02's end-to-end promotion gate;
- **semantic-only:** range semantics are useful but compact dispatch fails its cost gate, so the compact prototype is removed;
- **superseded:** current upstream/BigCherry execution removes the tiered-range requirement, with MET02/MET03 updated accordingly.

## Acceptance Criteria

- MET02 Phase A is implemented and built/tested before any permanent compact HIP path.
- No duplicate router, scheduler, placement store, MMQ J selector or HIP-autotune table is introduced.
- Any claimed speedup has correctness plus expected/observed active-work accounting.
- Failed experimental compact code is removed rather than left as unfinished production machinery.
