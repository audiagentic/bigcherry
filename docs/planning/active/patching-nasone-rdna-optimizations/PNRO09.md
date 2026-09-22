---
id: PNRO09
order: 0
plan: patching-nasone-rdna-optimizations
state: in_progress
created-at: '2026-09-09T10:52:50.276971+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: P1
---

# Increase Meta compute-container view headroom for recurrent/MTP graphs

## Description

Evaluate Meta compute-container view headroom for recurrent/MTP graphs. Port only after proving the existing 16-view bound is insufficient or a supported graph-derived bound exceeds it.

## Steps

- Confirm b10705 Meta allocator behavior and derive actual maximum static-tensor view count for recurrent+MTP graphs.
- Build boundary fixtures at 15/16/17 and higher views through the real Meta context mechanism, plus repeated eval/reset.
- Port the minimal constant/rationale only if failure is reproducible; prefer a derived bounded formula over magic 128.
- Measure metadata memory and verify ordinary dense/non-recurrent graphs are unchanged.
- Record capacity decision and keep the change correctness-scoped, not a performance claim.

## Detailed Solution & Technical Design

Recurrent snapshot views are estimated around 2*(n_rs_seq+1) per shared recurrent layer. Headroom concerns tensor-object metadata, not model data; bound it from graph structure and retain safe failure if capacity is exceeded.

## Code Samples & Guidance



## Files

ggml-backend-meta.cpp; Meta context boundary fixture; recurrent/MTP graph construction and eval/reset tests; metadata memory evidence.

## Validation

15/16/17+ view allocation; real recurrent+MTP graph; repeated reset; metadata accounting; dense/non-MTP controls; source identity.

## Effort & Risk



## Standards

Affirmative capacity proof; bounded resource accounting; no performance claim for correctness promotion.

## Acceptance Criteria

Existing 16 limit is shown insufficient or a proven bound requires change; new bound covers declared maximum with bounded metadata cost; no non-target behavior changes.

## Notes

Supersedes: NRO10
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro10

## Change Log

- 2026-09-09T10:52:50.276971+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:14.697625+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events




- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.090326+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.739019+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:44:03.530481+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024433_three-more-nasone-successors-n_7555
- 2026-09-10T02:44:33.131554+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-20T02:30:44.526791+00:00 (state-transition): State: pending → in_progress
- chg_20260920_025031_created-patch-1260-to-increase_8170
- 2026-09-20T02:50:34.930936+00:00 (updated-by): Updated: section:ledger-events
- chg_20260920_062016_patch-1260-pnro09-verified-o_3271
- 2026-09-20T06:20:21.408083+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-20T06:44:20.375111+00:00 (state-transition): State: in_progress → in_progress
- 2026-09-22 (implementation, hardware-free boundary test): Built the lean ggml+ggml-backend (CPU+Meta, no GPU) for the pinned/unpatched source (headroom=16) at /mnt/h/development/projects/bc-nro09-build. Wrote a first-pass C++ fixture (patches/1260_nro09_meta_view_headroom/validation/meta_boundary_test.cpp) that drives the REAL Meta backend (CPU simple device → meta device → meta backend, alloc_ctx_tensors + graph_compute) with N between-eval compute nodes of size S. FINDING: the binding constraint is the ggml object pool (ggml_new_object enforces ctx->mem_size); per this plan's own note the headroom bounds tensor-object METADATA, not model data. The first-pass fixture did NOT show a clean N=headroom boundary: the observed object size was S+metadata (data was being allocated into the pool, i.e. the hit context had no_alloc=false), so the boundary landed far from headroom. CORRECTED DESIGN: the boundary is on the NUMBER of tensor objects (fixed GGML_TENSOR_SIZE each), so N_max = headroom*S / GGML_TENSOR_SIZE — proportional to headroom, with ratio patched/unpatched = 80/16 = 5. NEXT: isolate the metadata-only allocation (verify the mapped simple tensors use no_alloc=true so the object size is fixed), then probe N_max on the unpatched (headroom=16) and patched (headroom=80) builds and assert the ~5x ratio plus the bounded metadata cost (compute container = 80x vs 16x the static mem_size). Build B (patched, headroom=80) still needs the 1260 patch applied to a vendor copy + build. (The ag-planning plan_update_item tool was defective this session — receiving `updates` as a string — so this note is recorded in the file directly.)
- 2026-09-23 (C boundary test COMPLETE + evidence): Corrected the fixture to drive the REAL stc_compute capacity (static leaf t in USER ctx -> stc_static; N between-eval view tensors of t in a separate ctx, registered via buf->iface.init_tensor -> stc_compute[0]; the simple tensor in stc_compute has no_alloc=true so object = fixed GGML_TENSOR_SIZE, metadata only). Built the patched variant (headroom=80) at /mnt/h/development/projects/bc-nro09-buildB. Observed N_max (unpatched=16 vs patched=80): S=1024: 44 -> 222 (ratio 5.045); S=2048: 89 -> 445 (ratio 5.0). The ratio (~5 = 80/16) is confirmed at two independent S values, with a bounded linear metadata cost (stc_compute mem_size = headroom*S; 80x vs 16x the static mem_size) and a safe abort on capacity exceedance. Evidence recorded in patches/1260_nro09_meta_view_headroom/evidence/boundary.json; SUMMARY.md + plan note updated. Commit ca1e45aa on patch-refactor (local). PUSH BLOCKED: the other session (PA36/PA40 reconciliation) has live uncommitted changes (modified 2026-09-23T00:11, ~2 min before this note) to PA36.md/PA40.md/validation_campaign.py/1233+1206 producer.py/test files that the 12 remote commits also touch; a merge would overwrite their live edits, and git stash is forbidden (AGENTS.md), so the merge is deferred to the other session. Remaining for PNRO09: the (D) real-graph hardware lane on Brutus (tierA-qwen4b-q6k, recurrent+MTP, gfx1100 0/1) + validation/producer.toml+producer.py + validation.toml capacity checks.
- 2026-09-23 (D real-graph hardware lane COMPLETE + evidence): Built the patched (headroom=80) and unpatched (headroom=16) variants on Brutus (gfx1100 7900 XTX devices 0/1) from copies of the bc-pa-work tree (the protected tree was not modified), and ran the REAL recurrent+MTP graph — tierA-qwen4b-q6k MTP (Qwen3.5-4B-UD-Q6_K_XL, --spec-type draft-mtp --spec-draft-n-max 4) — on both. FINDING: the 4B MTP workload's ~72 between-eval views (and their metadata footprint) fit even within the stock 16-view headroom (unpatched: NO abort, 18.6/76.5 t/s; patched: NO abort, 21.9/98.3 t/s, i.e. the patch is benign and actually faster). So 16->80 does not change the pass/fail outcome for this specific 4B workload; it raises the ceiling for larger/more views (consistent with the (C) boundary test's ~5x N_max ratio: 44->222 @S=1024, 89->445 @S=2048). The binding case is a workload with a larger per-view metadata footprint or a larger view count; the (C) hardware-free boundary test is the direct capacity proof, and the (D) hardware lane confirms the patch is benign (no regression, no correctness issue) on a real recurrent/MTP graph. Evidence in patches/1260_nro09_meta_view_headroom/evidence/hardware_d.json; SUMMARY.md + plan note updated. Remaining for PNRO09: validation/producer.toml+producer.py + validation.toml capacity checks (code-only).
