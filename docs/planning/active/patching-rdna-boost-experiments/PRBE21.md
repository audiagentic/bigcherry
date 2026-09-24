---
id: PRBE21
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:53.078402+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Fold SSM conv_input concat into qkv mmvq; rpb=2 for small-K MoE

## Description

TODO, two independent sub-candidates. (1) SSM conv_input-concat folding into qkv MMVQ (fork source 0510d7cfa) -- genuinely new fusion work, same family as PRBE18. (2) rpb=2 for small-K MoE -- this project ALREADY has the mechanism: mmvq.cu's calc_rows_per_block(ncols_dst, table_id, small_k, nwarps) has a small_k parameter (verified b11126) returning `small_k ? nwarps : 1` for ncols_dst==1 on GENERIC/GCN/TURING/GB10 tables, and the HI09 explicit-geometry template params (nwarps_explicit/rows_per_block_explicit, patch 0600_mmvq_geometry, state=validated core framework) already let a generated instance force rows_per_block=2 directly -- sub-candidate 2 is a catalog-driven candidate-enumeration task, not new kernel code.

## Steps

Sub-candidate 1 (conv_input fold):
1. Audit ggml/src/ggml-cuda/ssm-conv.cu/.cuh and ssm-scan.cu/.cuh for the current concat->conv1d->qkv-read chain.
2. Define guard: contiguous conv_state, F32, ne[1]==1 (decode shape); insert the fold at the same fusion-detection site used for PRBE18 (ggml-cuda.cu ggml_can_fuse_subgraph idiom, ~3300-3350) as a sibling branch, OR as a lower-level change inside ssm-conv.cu if the fold is below the op-fusion granularity (i.e. inside a single op's kernel rather than across ops) -- determine which during implementation by checking whether conv_input concat is its own ggml op (GGML_OP_CONCAT) feeding GGML_OP_SSM_CONV, or already an internal SSM_CONV detail.
3. Fallback to native concat+conv when guard fails.
Sub-candidate 2 (rpb=2 small-K MoE):
4. Do NOT hand-write a new kernel. Use tools/bigcherry/tuning/catalog.py's enumerate_mmvq() to add a small_k=true, rows_per_block=2 candidate row for the target quant types at small expert hidden-dim (K) shapes seen in tiny/A3B MoE configs, gated gfx1100 (or the relevant architecture family).
5. Run the existing record->tune->promote campaign (tools/bigcherry/campaign/campaign.py) scoped to this candidate set; do not assume cache or SSM-chain dependency (per item text) -- this sub-candidate is independent of PRBE05/PRBE18.
6. Compare against the native (small_k=false) row as control.

## Detailed Solution & Technical Design

Sub-candidate 1 needs new fusion-detection/kernel work (see PRBE18's plan for the analogous mechanism). Sub-candidate 2 is materially smaller than the item text implies once mapped onto the real framework: `calc_rows_per_block`'s `small_k` branch and the HI09 `rows_per_block_explicit` template parameter already exist and are validated infrastructure (patch 0600_mmvq_geometry, GROUP=core, STATE=validated) -- PRBE21's job for sub-candidate 2 is to define the right (type, shape-signature) candidate rows in the catalog and run the campaign, not to touch mmvq.cu's kernel body at all.

## Code Samples & Guidance

Real b11126 anchor (verified), ggml/src/ggml-cuda/mmvq.cu calc_rows_per_block:
```
if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
    switch (ncols_dst) {
        case 1:
            return small_k ? nwarps : 1;
        ...
```
Note MMVQ_PARAMETERS_RDNA3_0 (gfx1100's real table_id) and MMVQ_PARAMETERS_RDNA4 are NOT in this small_k-aware list -- their calc_rows_per_block falls through to the final `return 1;` regardless of small_k. This means gfx1100's native calc_rows_per_block never uses small_k today; sub-candidate 2's rows must be forced via the HI09 explicit template params (nwarps_explicit/rows_per_block_explicit), not by expecting small_k to already do anything on RDNA3. Patch package for sub-candidate 1: patches/12xx_rd27_ssm_conv_fold/patch.toml (state=untested, plan-item=RD27, requires=[] unless PRBE18's fusion-detection branch lands first, in which case list it). Sub-candidate 2 is expressed as new catalog.py candidate rows plus a campaign config, not a hand-authored patch package edit to mmvq.cu.

## Files

Sub-candidate 1: ggml/src/ggml-cuda/ssm-conv.cu/.cuh, ssm-scan.cu/.cuh, ggml-cuda.cu (fusion detection); patches/12xx_rd27_ssm_conv_fold/. Sub-candidate 2: tools/bigcherry/tuning/catalog.py (enumerate_mmvq), campaign config for the small-K MoE signature set, no source-file Edit.

## Validation

Sub-candidate 1: test-backend-ops SSM_CONV correctness, fused-vs-unfused equality, gfx1100 timing. Sub-candidate 2: `PYTHONPATH=tools python -m bigcherry patch-lint`/campaign dry-run on the new candidate rows, test-backend-ops MUL_MAT_ID small-K coverage, then the existing record->tune->promote->replay campaign (tools/bigcherry/patch/validation_campaign.py) on Brutus comparing small_k=2 vs native for the target MoE shapes -- not run here.

## Effort & Risk

Sub-candidate 1: M (new fusion + kernel work, similar risk profile to PRBE18). Sub-candidate 2: S (catalog-only, reuses validated HI09 infrastructure) -- the two should be tracked and possibly promoted independently since they have very different effort/risk.

## Standards

Dependency audit; exact pattern; memory safety; isolated qualification.

## Acceptance Criteria

Fused path is correct for exact SSM/small-K patterns, unsupported patterns fall back, and any benefit is shown in isolated causal evidence before composition.

## Notes

Supersedes: RD27
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd27

2026-09-24 relevance at b11126: TODO. Sub-candidate 2 is materially cheaper than the item text implies: verified that calc_rows_per_block's small_k parameter and the HI09 nwarps_explicit/rows_per_block_explicit template mechanism already exist as validated core framework (patch 0600_mmvq_geometry); gfx1100's RDNA3_0 table_id does not consume small_k natively, so the explicit-geometry override path is required. GPT design request submission for this item hit a rejected/queue-saturated gateway (concurrency limit reached: 8 queued, 2 running gateway-wide) and was not resubmitted successfully in-session; plan authored directly from verified source.

## Change Log

- 2026-09-09T10:54:53.078402+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:06.531616+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.222985+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.942391+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:51:58.669575+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025218_rdna-successors-prbe2022-now_5714
- 2026-09-10T02:52:18.419086+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:30:22.125041+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
