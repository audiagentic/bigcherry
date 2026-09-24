---
id: PRBE18
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:45.862958+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Evaluate SSM pre-scan chain fusion (conv + l2_norm pair + gate/beta)

## Description

TODO. Ports a 16-node SSM pre-scan chain fusion (conv+SiLU+Q/K-norm+V+gate/beta) as an isolated gfx1100 candidate. Relevance confirmed at b11126: upstream already has SSM_CONV fusion infrastructure (ggml-cuda.cu ggml_can_fuse_subgraph, two existing branches at ~3315 for SSM_CONV+SiLU and ~3330 for SSM_CONV+ADD+SiLU) but nothing beyond 3 nodes -- the 16-node chain (through Q/K normalization, V, gate/beta) is NOT upstream-absorbed. Depends on PRBE05 (patch 1235, q8_1 activation cache -- 4 references in fork source) and PRBE19 (post-fix source-state rule, applied below). GPT design request req_9d3d9188405f49f3 was submitted but the gpt-auto queue was saturated (8 queued/2 running gateway-wide) and did not return within this session; design below was produced directly against verified b11126 source and must be reviewed by GPT before a coding agent starts (resume via that request id or open a fresh dev-gpt-agent session).

## Steps

1. Verify PRBE05/patch 1235 (q8_1 activation cache) is available; note its API surface before wiring the fused SSM's downstream Q/K read through it.
2. Enumerate ALL 16 ops in order (CORRECTED -- required before any anchor work, was previously left unenumerated): SSM_CONV, UNARY(SiLU), then per branch the Q-norm pair (VIEW, RMS_NORM), K-norm pair (VIEW, RMS_NORM), V-view (VIEW), and gate/beta ops (their exact ggml_op/unary sequence must be read from the real graph-construction source for the target model at implementation time, not assumed) -- write the full ggml_op/unary-op list plus ggml_check_edges/out-nodes definition before touching ggml-cuda.cu.
3. Detection anchor (CORRECTED location): in ggml/src/ggml-cuda/ggml-cuda.cu's fusion-detection function, insert the new 16-op branch BEFORE the existing 2-node check `if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL` -- not after/inside the 3-node ADD branch as previously stated (that placement contradicted the intended insert_before semantics). Dispatch anchor is in `ggml_cuda_try_fuse` (ggml-cuda.cu:3432, confirmed real function; called from ggml-cuda.cu:4325): `if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU }))`-style call, extended for the 16-op pattern.
4. Guard predicate: F32 dtype on every node, ggml_is_contiguous on conv_state/qkv views, ne[1]==1 (decode shape only), shared alpha/beta tensor identity between the two norm ops, matching epsilon on both RMS_NORM ops, tracked conv_states producer, PLUS the required output-range/memory-safety check via ggml_check_edges and explicit output-node list (do not rely on op-sequence matching alone).
5. Choose ONE concrete fused kernel implementation site and file (CORRECTED -- previously left as an either/or with contradictory apply-path location): implement in ggml/src/ggml-cuda/ssm-conv.cu (extend the existing SSM_CONV kernel file) rather than a new file, since BigCherry's patcher edits existing files via anchored Edit()s; define the exact call/skip count for the fused launch replacing the 16 individual op launches.
6. Preserve wrong-wiring/no-fuse fallback: any guard failure emits the native unfused op sequence unchanged.
7. Run fused-vs-unfused numeric equality and graph-capture timing on gfx1100 before any composition with other patches.

## Detailed Solution & Technical Design

Extends upstream's existing `ggml_can_fuse_subgraph`-based SSM_CONV fusion mechanism (already fusing 2- and 3-node SSM_CONV+SiLU[+ADD] patterns at b11126) rather than inventing new fusion infrastructure. The 16-node candidate is a much longer chain through Q/K RMS-normalization (shared alpha/beta, matching epsilon), a V view, and gate/beta -- none of which upstream's existing branches recognize. Source the fused kernel body from PRBE19's reviewed post-fix state (provenance 9e46e1fd..., verified equivalent to v3 snapshot c8af5361...), using the CURRENT selector-derived MMVQ launch geometry (mmvq.cu's calc_nwarps/calc_rows_per_block, or the HI09 explicit-geometry template params already in mmvq.cu) rather than any hardcoded historical warp count from the original fork commit -- this is the concrete meaning of PRBE19's bake-in rule for this item. The four q8_1_cache references in the historical source are the real evidence for the PRBE05/patch-1235 dependency; wire the fused kernel's activation read through that cache API rather than importing a second cache implementation.

## Code Samples & Guidance

Real b11126 anchor (verified): ggml/src/ggml-cuda/ggml-cuda.cu, function containing:
```
if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
 && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
    const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
    const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
    ...
    return true;
}
```
Insert a new `if (ops.size() == 16 && ...)` branch immediately after the existing 3-node ADD+SiLU branch (ends ~line 3346), following the same struct: extract each `cgraph->nodes[node_idx+k]`, type-check F32, contiguity-check, then the SSM-specific checks (shared alpha/beta identity, matching epsilon, ne[1]==1, tracked conv_states producer) before returning true. Patch package sketch: patches/12xx_rd24_ssm_prescan_fusion/patch.toml (id, order in the 12xx range after existing SSM/MMVQ patches, state="untested", kind="enhancement", plan-item="RD24", requires=["1235_rd09_q81_activation_cache_foundation"]); patch.py using `from bigcherry.patcher import Edit, FilePatch` with an `Edit(id="ssm_prescan_fuse_detect", anchor=r"if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD", mode="insert_before", text=<new 16-node branch>, guard=r"GGML_OP_SSM_CONV")` plus a second Edit adding the fused kernel file and its dispatch call. Read patches/1204_rd08_q6k_mmvq_vdr2/ first for exact FilePatch/Edit conventions before authoring.

## Files

ggml/src/ggml-cuda/ggml-cuda.cu (fusion detection ~3300-3350, applied-fusion dispatch ~3200-3260); ggml/src/ggml-cuda/ssm-conv.cu, ssm-conv.cuh, ssm-scan.cu, ssm-scan.cuh (new fused kernel); patches/12xx_rd24_ssm_prescan_fusion/{patch.toml,patch.py,SUMMARY.md}; new test-backend-ops SSM_CONV/RMS_NORM fused case; PRBE05/patch-1235 cache API.

## Validation

1. `PYTHONPATH=tools python -m bigcherry patch-lint` on the new package. 2. `patch-rebase-check --focal-overlay 12xx_rd24_ssm_prescan_fusion --source bigcherry-tuning`. 3. New/extended test-backend-ops case comparing fused vs unfused output bit-for-bit on the exact guarded shape, plus a wrong-wiring fixture (untracked conv_states) proving fallback engages. 4. PRBE19 bake-in check: diff the fused kernel's sourced SSM region against provenance 9e46e1fd... and confirm semantic equivalence to v3 snapshot c8af5361..., not the raw historical diff. 5. Hardware (Brutus, not run here): gfx1100 graph-capture + causal timing via tools/bigcherry/patch/validation_campaign.py, fused vs unfused arms.

## Effort & Risk

L (large): new CUDA kernel authoring plus fusion-detection extension in a hot dispatch path; correctness risk is high (silent wrong-wiring on a mismatched SSM shape) but is bounded by the guard/fallback design and by reusing upstream's already-proven ggml_can_fuse_subgraph idiom rather than a new mechanism.

## Standards

Exact graph pattern; Wave-2 dependency; no duplicate superseded ports; fallback.

## Acceptance Criteria

Exact 16-node pattern is correct and captures; unsupported graphs do not fuse; only a dependency-complete arm may be promoted on positive causal evidence.

## Notes

Supersedes: RD24
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd24

Supersedes: RD24 (closed historical predecessor); RD14 and RD16 are closed/superseded historical designs and are not separate ports. Preserve RD24 source commit 4a4da30e... as provenance. Live dependencies are PRBE05 and PRBE19.

2026-09-24 relevance at b11126: TODO. Confirmed via grep of b11126 ggml-cuda.cu: existing SSM_CONV fusion covers only 2-3 nodes (SiLU, optional ADD-bias); the 16-node candidate is not upstream. GPT design request req_9d3d9188405f49f3 (dev-gpt-agent, gpt-auto) submitted but queue-saturated/no response in-session -- plan authored directly from verified source, needs GPT/human review before coding starts.

2026-09-24 GPT review req_2b717df095b44703 applied: enumerated the required full 16-op/edge/output-node definition (was previously unenumerated); corrected the detection-anchor location to insert before the 2-node UNARY+MUL check (was contradictorily described); corrected the dispatch anchor to the verified real function ggml_cuda_try_fuse (ggml-cuda.cu:3432); resolved the ambiguous kernel-location choice to ssm-conv.cu (existing file, patcher is anchor-only).

## Change Log

- 2026-09-09T10:54:45.862958+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:53.591511+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.209644+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.922415+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:50:30.902295+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025049_rdna-successors-prbe1719-now_5726
- 2026-09-10T02:50:49.219544+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T09:52:30.160133+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:notes
- chg_20260912_095506_cleaned-the-active-rdna-boost_4906
- 2026-09-12T09:55:06.240435+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T10:09:39.929105+00:00 (updated-by): Updated: section:validation
- chg_20260912_101015_fixed-the-remaining-plan-taxon_2574
- 2026-09-12T10:10:15.548038+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:29:21.076816+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:41:50.202769+00:00 (updated-by): Updated: section:steps, section:notes
