---
id: FKE01
order: 1
plan: patching-fused-kernel-equivalence
state: pending
created-at: '2026-10-06T15:29:25.180488+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Validate every fused GPU kernel against its unfused path (bit-exactness, fidelity, speed)

## Description

Fused launches are on by default in the production build, and which candidates fuse depends on real memory addresses (ggml_cuda_check_fusion_memory_ranges refuses a fusion whose output overlaps a live input). MSM02 showed the consequence: changing only the buffer layout changed about 45 of ~19,000 fusion decisions and the greedy text, while GGML_CUDA_DISABLE_FUSION=1 made the two layouts identical. So at least one fused kernel is not bit-identical to the unfused sequence it replaces, and the output of production depends on allocator layout. We do not know which fusions are bit-exact, how far the inexact ones are from the unfused path and from the CPU f32 reference, or whether each one is actually faster.

## Steps

1. Whole-set measurement on production (Flash-Next, 3-GPU tensor split): fusion on vs GGML_CUDA_DISABLE_FUSION=1, ABBA prefill / decode at 8K and 98K, probes of each arm against the other and against the CPU f32 reference (which arm is closer to the reference). 2. Inventory: list every fusion in the production tree (upstream ggml_cuda_try_fuse / can_fuse paths and our patches 1205, 1206, 1237, 1241, 1274, 1309-1313 and later) with its off switch, or add a per-fusion mask (one env bitmask) where none exists. 3. Per fusion: an op-level test in the style of tests/test-mul-mat-id-range.cpp - same inputs through the fused launch and through the unfused node sequence on each GPU arch (gfx1100, gfx1201, gfx1030): bit-exact yes/no, max abs / rel difference, both against the CPU reference. 4. Per fusion on hardware: speed with only that fusion off (ABBA). 5. Classify: bit-exact (keep), inexact but closer to or as close as unfused to the reference and faster (keep, record the tolerance), inexact and no faster or further from the reference (fix or remove). 6. Make fusion selection independent of buffer layout where cheap (e.g. decide from graph structure and let the kernel handle overlap, or make the allocator keep fusion outputs off live inputs), so output no longer depends on allocator decisions.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

vendor ggml-cuda/ggml-cuda.cu (fusion predicates, read), patches/12xx-13xx fusion packages, patches/1339_meta_memory_report (fusion_overlap counters), tools/lab/flash-next/queue-env-ab.sh + flash-fidelity.sh, new tests/test-fused-equivalence.cpp via a patch package

## Validation

A table per fusion: bit-exact or max difference, distance to the CPU reference with and without it, speed delta, decision. Production output identical across two buffer layouts (common arena vs 1340) once inexact fusions are fixed or selection is layout-independent.

## Effort & Risk

Low risk (measurement and tests first). The op-level harness is the main work; per-fusion off switches may need small edits inside the owning patches.

## Standards



## Acceptance Criteria



## Notes

Owner question 2026-10-07: 'Should we validate the fused kernels against unfused'. Does not invalidate the promoted fusion patches' proofs (their evidence was taken with the path on); this adds an equivalence check that was never part of those proofs. Unblocks an identical-output gate for memory-layout changes (MSM02).

2026-10-07 STEP 1, whole-set measurement (production Flash-Next, 3-GPU tensor split, ctx 49152, build b-metamem-b11402h2 with the 1340 flag off; ABBA, A = fusion on (production), B = GGML_CUDA_DISABLE_FUSION=1; run fke01-a). 8K depth: prefill 1072.2/1065.1 vs 1032.1/1026.7 (+4%); decode 86.3/85.9 vs 78.5/78.6 (+10%); acceptance 349/485, 348/488 vs 354/468. 24K depth: prefill 1073.5/1066.8 vs 1027.1/1028.2 (+4%); decode 76.4/76.7 vs 64.7/66.0 (+17%); acceptance 346/495 vs 339/513, 341/507. Probes at 8K (24 probes, C = CPU f32 reference): fused vs reference top-1 21/24, TV mean 0.0957; unfused vs reference top-1 22/24, TV mean 0.0784; fused vs unfused top-1 23/24, TV mean 0.0827 max 0.407; fused against itself 0.0000. READING: fusion as a set is clearly worth its speed (+4% prefill, +10..17% decode, complete separation). It is not bit-exact against the unfused path (TV 0.083), and on this probe set the unfused path is slightly closer to the CPU reference (0.078 vs 0.096, 22 vs 21 top-1) - a small gap on 24 probes, not a quality verdict, but it means the fused set carries a numeric cost that has never been attributed to individual fusions. With fusion off the overlap counter is silent (no checks), with it on 1107 of 19730 candidates were refused for overlap in this run. Next: inventory + per-fusion off switches, op-level equivalence test, per-fusion speed.

2026-10-07 STEP 2, bisect by first-node op (1342, build b-fusebis-b11402a3, production, ctx 49152, 8K depth, 256 decoded tokens, ABBA per family, A = all fusion on, B = that family off). FUSIONS TAKEN in one request with everything on (count / nodes elided): RMS_NORM 23668 / 23668; SCALE 18624 / 27914; UNARY 4513 / 4513; SOFT_MAX 4249 / 38241; MUL 4021 / 76399; GATED_DELTA_NET 3456 / 10368; SSM_CONV 3456 / 3456; MUL_MAT_ID 1745 / 3541; ADD 624 / 4368; MUL_MAT 291 / 582 (ROPE: none taken). OUTPUT with the family off, against production: IDENTICAL greedy text for MUL_MAT_ID, MUL_MAT, RMS_NORM, UNARY, SCALE, ROPE, SSM_CONV, ADD, GATED_DELTA_NET; DIFFERENT for MUL (md5 a30fef8d) and for SOFT_MAX (md5 ff478c34). So nine of the eleven families are bit-exact against their unfused sequences on this workload, and the two that are not are the long chains: MUL-start (19 nodes elided per fusion - the MoE weighted expert reduction, 2k+1 nodes) and SOFT_MAX-start (9 nodes per fusion - the top-k MoE routing block). SPEED with the family off (prefill A vs B; decode A vs B): MUL_MAT_ID level (first pair cold); MUL_MAT 1066/1072 vs 1070/1062, 83.9/84.1 vs 85.8/85.2; RMS_NORM 1073.9/1064.1 vs 1058.4/1057.0 (-1%), 84.2/83.9 vs 82.8/82.6 (-1.5%); UNARY level; SCALE level, decode 83.8/83.5 vs 83.2/82.2; ROPE level; SSM_CONV 1075/1076 vs 1070/1070, decode 84.5/85.2 vs 83.9/83.7 (-1%); ADD decode 84.6/84.9 vs 83.8/83.4 (-1%); GATED_DELTA_NET 1076/1075 vs 1068/1066 (-1%), decode 84.0/84.6 vs 83.1/83.1 (-1.5%); MUL 1076.6/1071.5 vs 1062.9/1062.2 (-1%), decode level (84.0 vs 83.6/84.5); SOFT_MAX prefill 1074.5/1067.8 vs 1066.8/1066.0, decode 84.3/83.8 vs 81.4/79.6 (-4%). No single family explains the +10..17% decode of the whole set (fke01-a): the gain is the sum of many small ones, the largest single one being the top-k routing fusion. NEXT: op-level equivalence tests for the two inexact chains (what exactly differs: accumulation order in the weighted reduction; the top-k block's softmax / selection arithmetic), and decide per chain: make it bit-exact, or record the tolerance. A layout-independent selection for these two would remove the output dependence on the allocator (MSM02).


2026-10-08 b11474 rebase: upstream e117148a4 ("CUDA: make the alloc_deps check batch independent") changed shared-expert fusion admission by replacing the batch-dependent `ggml_cuda_should_fuse_mul_mat_vec_q(up)` test inside `ggml_cuda_match_shared_expert` with the batch-independent prerequisite `ggml_is_quantized(up->src[0]->type)`, then applying `ggml_cuda_should_fuse_mul_mat_vec_q(cgraph->nodes[i + 2]->src[1])` later in `ggml_cuda_try_fuse`. This keeps graph-optimizer/alloc-deps topology stable across ubatch sizes and avoids re-reserve/scheduler synchronization caused by a batch-size-dependent optimized graph. It does **not** remove or weaken the runtime `ggml_cuda_check_fusion_memory_ranges` address-overlap gate: actual tensor buffer addresses can still accept/refuse a fusion. Therefore e117148a4 reduces batch-size-dependent topology/layout churn but **leaves the allocator-address dependence measured by MSM02 intact**.

## Change Log

- 2026-10-06T15:29:25.180488+00:00 (created-by): Created by agent
- 2026-10-06T15:54:58.645699+00:00 (updated-by): Updated: section:notes
- 2026-10-06T17:18:14.568307+00:00 (updated-by): Updated: section:notes
