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

## 2026-10-15 authoritative narrow audit — SOFT_MAX-start MoE top-k fusion (BCOP217)

**Disposition:** FKE01 stays pending, but its next single slice is ordered expert-routing equivalence, not another global fusion framework. The historical "nine families bit-exact" claim is unsupported: the 2026-10-07 bisect compared *greedy text*, not tensor bits, logits, ordered IDs or MTP acceptance. SOFT_MAX-off changed greedy text, but the first divergent numerical producer remains unknown. Preserve validated 1342 and production fusion defaults.

### Pinned b11552 source, graph, lifetime and dispatch

- `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_try_fuse` (~3675–3747) starts MoE top-k matching at SOFT_MAX, UNARY or ARGSORT. `ggml_cuda_topk_moe_fusion` (~2976+), `ggml_can_fuse_subgraph`, `ggml_cuda_should_use_topk_moe` and `ggml_cuda_check_fusion_memory_ranges` admit SOFT_MAX→RESHAPE→ARGSORT→VIEW→GET_ROWS with optional bias, norm and scale; the fused operation must publish **both** ordered expert IDs and aligned weights. Delayed-softmax ARGSORT and sigmoid/sqrt-softplus UNARY routes are separate controls.
- `ggml/src/ggml-cuda/topk-moe.cu::{softmax_warp_inplace,topk_moe_cuda,ggml_cuda_op_topk_moe}` performs F32 exp/warp reduction, iterative top-k with lower-ID tie breaking, optional renormalization and scaling. `TOPK_MOE_ROWS_PER_BLOCK=8`. The kernel reads logits, synchronizes a block, then writes IDs/weights. The memory-range guard permits logits/output alias only for <=8 rows; for other cases the ordinary overlap rejection applies. Never disable this safety gate merely to make fusion layout-independent. A changed accumulation order is a **hypothesis**, not the proven cause of FKE01 output differences.
- **Concrete test-coverage defect:** pinned `tests/test-backend-ops.cpp::test_topk_moe` (~7271–7368) already constructs both outputs and exercises expert counts, 8/9/22 rows, norm, bias and scale. But its `err()` independently **sorts both arrays before NMSE**, so it cannot verify ordered expert IDs, ID→weight pairing or bitwise F32 weights. Two executions can pass sorted comparisons while assigning the same weight to different experts. Extend this existing test/compare callback rather than creating a second harness.
- `engines/llamacpp/patches/1342_fusion_bisect/patch.py`: `BIGCHERRY_FUSION_SKIP_OPS=SOFT_MAX` rejects every candidate *beginning* with SOFT_MAX; the taken/elided counters do not identify a specific top-k mode, source tensors, or matched fused/unfused call. Skipping a family can enable downstream alternatives (1342's own MUL-off receipt changes ADD/SSM_CONV counts). A flag/counter alone does not prove the executed route.
- **Historical BigCherry evidence:** b11402 whole-set fusion ON vs OFF measured ~+4% prefill and +10–17% decode; the SOFT_MAX-family-off arm lost ~4% decode. FKE01's 24 CPU-F32 probes slightly favored the unfused whole set. Different controls, source pins and output paths cannot be pooled or added. MSM02/1340's allocator-dependent fusion selection is an observed interaction, not proof that allocator memory is corrupt or that all other fusion families are bit-exact.

### Bounded discriminator, correctness and performance gates

1. Reuse `test_topk_moe` with **saved deterministic logits** and the existing 1342 switch; run fusion ON and SOFT_MAX-skip in separate processes because the env gate is static. Preserve the exact input bytes, bias, shape, top-k, norm/scale, graph/alias class, backend UUID, build/patch digest and launch trace. Require positive `topk_moe_cuda` activation and an unfused graph with no replacement fusion. Otherwise terminal `NO_CORRELATED_ROUTE`.
2. Compare `ids[token,k]` as ordered integers and `weights[token,k]` **paired with the same k**, reporting raw F32 bits, ULP, max absolute/relative, KLD/TV and reference distance. Require expected expert count, no duplicate/missing IDs, finite/tie/NaN policy and selected-weight normalization. Add a negative fixture where independently sorted IDs/weights match but expert→weight pairing differs. Identical greedy text is not bit identity.
3. Minimum matrix: nrows `{1,4,8,9,22}`, experts `{32,64,128}` plus 288 support control, top-k `{2,8}`, near/exact ties, extreme logits, -Inf and documented NaNs, norm/bias/scale, alias and nonalias, 2–6 MTP verify rows, repeated same-process requests, multi-ubatch, graph replay/reallocation and context shifts. Require actual physical gfx1100/gfx1201 route evidence; gfx1030 is conditional only. Hold source/graph identity across common vs 1340 arena.
4. Any finite non-tied ID/order mismatch, lost pair, skipped work, unsafe alias or graph fault => `ROUTER_CORRECTNESS_FAIL`; isolate first divergent op before a kernel edit. Identical IDs but different F32 weights => `NUMERICALLY_DIFFERENT`, quantify against fixed CPU-F32 and decide bounded math correction or explicit tolerance. Fully matched ordered pairs => `EQUIVALENT`. Do not globally disable fusion.
5. Only after correlation/correctness, paired current-pin fixed-work ABBA at ~8K/~98K with MTP/no-MTP, complete physical-rank critical-path and kernel/launch accounting. A pure speed change needs CI95-low >=3% E2E and <=1% unaffected-lane regression; a correctness fix can proceed without a gain but requires numerical improvement and bounded performance regression. Reject another scheduler, allocator, selector, cache, telemetry service or hardware queue.

**Upstream/fork:** llama.cpp [#27621](https://github.com/ggml-org/llama.cpp/pull/27621) already introduced multi-token top-k fusion and the eight-row barrier/alias exception (present in b11552); #30441 on 2026-10-13 only removes dead breaks. [SGLang #36811](https://github.com/sgl-project/sglang/pull/36811), merged 2026-09-01, removes a zero-bias tensor via compile-time `HAS_BIAS` and adds exact ordered weight/ID comparisons at M=1/8/32. Native ggml already has `topk_moe_cuda<n_experts,has_bias>`: adopt **the test discipline, not the Triton implementation or external hardware gains**. [vLLM fused_topk_router](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py) is a scoring/renormalization reference, not an AMD performance port.

**Ownership / 12-hour exclusion:** FKE01 owns cross-path numerical proof; 1342 is existing diagnostics; MSM02/1340 owns allocation; MUL-start weighted reduction and RD17/1207 remain separate fusion candidates. Do not modify current RAD02/03/05/06 Radiance helper, routing, mailbox, GEMV, prefill or queued lanes; RAD04 prefill remains separate. Last independent FKE01 update 2026-10-07 15:37Z; 1342 patch update 2026-10-09 05:37Z; no independent same-capability activity/PR/queued evidence was found in the prior 12 hours. Recent BCOP208–216 cover different slices.

**Host-only discriminator (this audit):** 1024 deterministic 64-expert F32 cases comparing illustrative sequential and lane-tree softmax reductions; 734 had differing weight bits with identical top-8 expert IDs. Independently sorting IDs and weights hid a changed expert→weight map; identical argmax hid changed logit bits. This is **not** a GPU result or proof of the historical cause. Python compilation passed; no repository pytest, patch application, HIP build, GPU execution or new inference benchmark.

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

2026-10-07 STEP 2, bisect by first-node op (1342, build b-fusebis-b11402a3, production, ctx 49152, 8K depth, 256 decoded tokens, ABBA per family, A = all fusion on, B = that family off). FUSIONS TAKEN in one request with everything on (count / nodes elided): RMS_NORM 23668 / 23668; SCALE 18624 / 27914; UNARY 4513 / 4513; SOFT_MAX 4249 / 38241; MUL 4021 / 76399; GATED_DELTA_NET 3456 / 10368; SSM_CONV 3456 / 3456; MUL_MAT_ID 1745 / 3541; ADD 624 / 4368; MUL_MAT 291 / 582 (ROPE: none taken). OUTPUT with the family off, against production: IDENTICAL greedy text for MUL_MAT_ID, MUL_MAT, RMS_NORM, UNARY, SCALE, ROPE, SSM_CONV, ADD, GATED_DELTA_NET; DIFFERENT for MUL (md5 a30fef8d) and for SOFT_MAX (md5 ff478c34). So nine of the eleven families produced identical greedy text in this workload; tensor, weight and expert-ID bit identity was not measured. The two families with different greedy text are the long chains: MUL-start (19 nodes elided per fusion - the MoE weighted expert reduction, 2k+1 nodes) and SOFT_MAX-start (9 nodes per fusion - the top-k MoE routing block). SPEED with the family off (prefill A vs B; decode A vs B): MUL_MAT_ID level (first pair cold); MUL_MAT 1066/1072 vs 1070/1062, 83.9/84.1 vs 85.8/85.2; RMS_NORM 1073.9/1064.1 vs 1058.4/1057.0 (-1%), 84.2/83.9 vs 82.8/82.6 (-1.5%); UNARY level; SCALE level, decode 83.8/83.5 vs 83.2/82.2; ROPE level; SSM_CONV 1075/1076 vs 1070/1070, decode 84.5/85.2 vs 83.9/83.7 (-1%); ADD decode 84.6/84.9 vs 83.8/83.4 (-1%); GATED_DELTA_NET 1076/1075 vs 1068/1066 (-1%), decode 84.0/84.6 vs 83.1/83.1 (-1.5%); MUL 1076.6/1071.5 vs 1062.9/1062.2 (-1%), decode level (84.0 vs 83.6/84.5); SOFT_MAX prefill 1074.5/1067.8 vs 1066.8/1066.0, decode 84.3/83.8 vs 81.4/79.6 (-4%). No single family explains the +10..17% decode of the whole set (fke01-a): the gain is the sum of many small ones, the largest single one being the top-k routing fusion. NEXT: op-level equivalence tests for the two inexact chains (what exactly differs: accumulation order in the weighted reduction; the top-k block's softmax / selection arithmetic), and decide per chain: make it bit-exact, or record the tolerance. A layout-independent selection for these two would remove the output dependence on the allocator (MSM02).


2026-10-08 b11474 rebase: upstream e117148a4 ("CUDA: make the alloc_deps check batch independent") changed shared-expert fusion admission by replacing the batch-dependent `ggml_cuda_should_fuse_mul_mat_vec_q(up)` test inside `ggml_cuda_match_shared_expert` with the batch-independent prerequisite `ggml_is_quantized(up->src[0]->type)`, then applying `ggml_cuda_should_fuse_mul_mat_vec_q(cgraph->nodes[i + 2]->src[1])` later in `ggml_cuda_try_fuse`. This keeps graph-optimizer/alloc-deps topology stable across ubatch sizes and avoids re-reserve/scheduler synchronization caused by a batch-size-dependent optimized graph. It does **not** remove or weaken the runtime `ggml_cuda_check_fusion_memory_ranges` address-overlap gate: actual tensor buffer addresses can still accept/refuse a fusion. Therefore e117148a4 reduces batch-size-dependent topology/layout churn but **leaves the allocator-address dependence measured by MSM02 intact**.

## Change Log

- 2026-10-06T15:29:25.180488+00:00 (created-by): Created by agent
- 2026-10-06T15:54:58.645699+00:00 (updated-by): Updated: section:notes
- 2026-10-06T17:18:14.568307+00:00 (updated-by): Updated: section:notes
