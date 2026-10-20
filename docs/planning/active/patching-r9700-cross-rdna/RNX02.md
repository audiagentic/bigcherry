---
id: RNX02
order: 2
plan: patching-r9700-cross-rdna
state: pending
created-at: '2026-10-03T01:32:50.179535+00:00'
breadth: ''
skill: advanced
created-by: codex
work: M
priority: P1
---

# R9X02 — Decode attention and QSA/indexer extraction

## Description

Extract the r9700-stack head-dim-256 decode/paged-attention and sparse QSA/indexer ideas for Qwen4Exp without duplicating existing FlashAttention tile tuning. Dense attention and QSA score/top-k remain separate acceptance units unless profiling proves fusion is required.

## Steps

- Review 1270, 1202, 1203, 1266, and 1271 before touching FlashAttention.
- Map Qwen4Exp graph construction to exact CUDA/HIP score, top-k/index-gather, and sparse-attention operations.
- Profile D=256 decode and QSA independently, including shapes, types, KV layout, token counts, MTP path, launch count, and bytes.
- Implement 1300 only as exact-shape/capability dispatch around the existing attention path; implement 1301 only for an independently measured QSA hot path.
- Keep KV numerical-format experiments independently bisectable.

## Detailed Solution & Technical Design

Create one default-off package-only experiment at ggml/src/ggml-cuda/fattn.cu::ggml_cuda_get_best_fattn_kernel(). When Q has D=256 and one query token, K/V are Q8_0, the existing D=256 Q8 vector case is available, and the architecture is RDNA3/RDNA4, return the existing BEST_FATTN_KERNEL_VEC path under an explicit opt-in flag. Do not add a kernel or dequantizer. Preserve the current selector for all unmatched cases and leave gfx103x unchanged initially.

## Code Samples & Guidance



## Files

- kernels/r9k_attn.hip
- kernels/r9k_qsa.hip
- kernels/r9k_qsa_score.hip
- r9700_vllm/kernels/attn.py
- r9700_vllm/attn/qsa.py
- src/models/qwen4exp.cpp
- ggml/src/ggml-cuda/fattn-tile.cuh

- patches/1300_rnx02_q8kv_vec_decode/patch.toml
- patches/1300_rnx02_q8kv_vec_decode/patch.py
- patches/1300_rnx02_q8kv_vec_decode/SUMMARY.md
- tools/tests/patch/test_1300_rnx02_q8kv_vec_decode.py

## Validation

Mechanics: exact positive and negative selector predicates, including non-Q8 types, non-D=256, multi-token Q, unavailable vector case, and non-RDNA3/4. Later hardware validation must cover n_tokens={1,2,4}, GQA variants, context lengths around padding/256 boundaries, backend-reference parity, graph replay, greedy/KLD checks, and unprofiled 80K E2E decode on gfx1100/gfx1201. Promotion requires measured E2E crossover evidence, not a kernel microbenchmark alone.

## Effort & Risk



## Standards

RDNA4 may use source geometry after semantic parity; RDNA3 retunes existing gfx11 primitives; RDNA2 ports only architecture-neutral ideas without gfx12 WMMA.

## Acceptance Criteria

The experiment is default-off and activated only by an explicit opt-in; it selects the pre-existing Q8 D=256 vector implementation only for the exact supported predicate; no new kernel, dequantizer, or staging path is introduced; unmatched selector behavior is unchanged; numerical and end-to-end evidence is recorded before any lifecycle promotion.

## Notes

Original source alias is R9X02. Proposed slots 1300 and 1301; 1201/1202/1203/1266/1270/1271 remain overlap owners.

Verbatim legacy source retained during R9X→RNX migration:

# R9X02 — Decode attention and QSA/indexer extraction

Status: planned
Proposed patches: `1300_r9x_attn_d256_decode`, `1301_r9x_qsa_score_indexer`
Depends on: R9X01
External source: `kernels/r9k_attn.hip`, `kernels/r9k_qsa.hip`, `kernels/r9k_qsa_score.hip`, `r9700_vllm/kernels/attn.py`, `r9700_vllm/attn/qsa.py`
Existing BigCherry owners: 1201, 1202, 1203, 1266, 1270, 1271



GPT design pass (req_83e7cdc000be4b9e): existing Q8_0/D=256 vector attention is already instantiated in fattn-vec.cuh, while the competing MMA/TILE path can stage the full K/V tensors. Recommended advisory package: 1300_rnx02_q8kv_vec_decode. This is a dispatch experiment, not a validation claim; quantify both staging removal and numerical/E2E tradeoffs.



Implementation update (2026-10-03): created advisory package 1300_rnx02_q8kv_vec_decode. It is untested, default-off, and deliberately not added to a recipe. The package targets the verified c061df198 fattn.cu selector anchor and uses the existing Q8_0/D=256 vector implementation.



Review RV4206 assessment: the QSA path has a separate cross-start greedy determinism defect in tied radix TOP_K selection; keep that as a distinct 1301 analysis/repair and do not fold it into 1300. The 1300 experiment remains justified by the measured verify-width q8_0->f16 attention cost, but requires backend parity and deterministic output checks.

## Goal

Extract the r9700-stack head-dim-256 decode/paged-attention and sparse QSA/indexer ideas that reduce launch/memory cost in Qwen4Exp, without duplicating BigCherry's existing FlashAttention tile tuning. Treat dense attention and QSA score/top-k as separate acceptance units unless profiling proves they must be fused together.

## BigCherry/llama.cpp mapping

Review `patches/1270_pnro14_rdna35_fa_tile_d256/patch.py` first: its exact target is `ggml/src/ggml-cuda/fattn-tile.cuh` and it already owns a gfx1151 D=256/ncols=32 tile row. Also review `1202_rd04_bf16_flash_attn_tile`, `1203_rd050607_rdna4_wmma_fa_q6k_mmq`, `1266_rd05_wmma_fa_tileq_sync`, and `1271_prbe54_q5_kv_dequant_f16` before touching FlashAttention.

For the QSA/indexer path, map graph construction in `src/models/qwen4exp.cpp` to the exact CUDA/HIP ops that implement score calculation, top-k/index gather, and sparse attention. If the operation lowers through generic `mul_mat`/top-k rather than a named QSA backend file, patch the narrow dispatch/fusion point; do not introduce a new op solely to mirror vLLM naming.

## Implementation sequence

1. Profile D=256 decode and QSA score/index selection independently. Capture shapes, types, KV layout, token counts, and whether MTP verify takes the same path.
2. Diff r9k attention/QSA algorithms against current `fattn-*` implementation. Extract only novel dataflow: paged-KV access pattern, score staging, reduction/top-k fusion, fewer materialized intermediates, or launch-count reduction.
3. `1300`: add an exact-shape/capability dispatch around the existing attention implementation. Keep generic/tile-table behavior unchanged outside the predicate.
4. `1301`: only if QSA has an independently measurable hot path, add the sparse score/indexer optimization with a standalone disable/trace path.
5. Never fold a KV numerical-format experiment into these patches; 1271 and KV quantization work must remain independently bisectable.

## RDNA adaptation

| Generation | Plan |
|---|---|
| RDNA4/gfx12 | Use r9k geometry/instruction sequence as the reference, after parity with llama.cpp semantics. |
| RDNA3/gfx11 | Reuse paging/staging/fusion. Retune block/wave layout; use existing gfx11 FA primitives instead of gfx12-only WMMA. Test gfx1100 and gfx115x separately where dispatch differs. |
| RDNA2/gfx103x | Port only memory-layout/launch-reduction ideas that survive without gfx12 WMMA. Prefer existing RDNA2 attention math primitive and wave configuration. |

## Change Log

- 2026-10-03T01:32:50.179535+00:00 (created-by): Created by codex
- 2026-10-03T01:38:28.533176+00:00 (updated-by): Updated: section:notes
- 2026-10-03T02:18:04.994678+00:00 (updated-by): Updated: section:detailed_solution, section:validation, section:acceptance_criteria, section:notes

## Reviews

- RV4206
- 2026-10-03T02:21:03.292673+00:00 (updated-by): Updated: section:files, section:notes
- RV4213

## Ledger-events




- chg_20261003_022112_added-a-safe-opt-in-q8-attent_5492
- 2026-10-03T02:21:15.702342+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-03T02:23:15.628294+00:00 (updated-by): Updated: section:notes
- chg_20261003_040732_long-context-qwen4exp-output-i_9401
- 2026-10-03T04:07:36.135523+00:00 (updated-by): Updated: section:ledger-events
- chg_20261003_084248_investigated-and-rejected-a-fl_8872
- 2026-10-03T08:42:51.875082+00:00 (updated-by): Updated: section:ledger-events
- chg_20261003_104706_faster-long-context-decoding-f_1057
- 2026-10-03T10:47:17.743572+00:00 (updated-by): Updated: section:ledger-events


## 2026-10-07 optimisation audit — quantized-KV vector attention ownership

### Disposition

Patch 1298 already falsified the broad "keep MTP verify on the Q8 vector kernel" hypothesis: at ~30K cached with MTP3 it measured 58.9 ms/step versus 56.1/54.3 ms baseline (~6% slower). The cause recorded in 1298 is structural: the vector path lacks GQA sharing and re-reads quantized KV for verify columns, while the tile path amortizes one conversion/read across the GQA group. Do not revive a Q>2 selector override without new kernel-level evidence.

Patch 1300 is a narrower n_tokens=1 experiment and remains untested. Its next gate must not be a selector-only A/B. Current upstream RDNA4 evidence (llama.cpp issue #27796, 2026-08-27) measured Q8_0 KV slower than F16 at D=256 and found the quantized vector kernel uses a flat `nthreads_KQ_q = 2`; the issue explicitly labels under-parallelization as a hypothesis, not a proven cause. That mechanism is directly testable on BigCherry and is upstream to 1300's selector decision.

### Ownership and implementation gate

RNX02 remains the sole owner of D=256 Q8 decode-attention extraction. Do not add a second FA dispatcher or KV staging mechanism. Reuse `ggml_cuda_get_best_fattn_kernel()`, the existing Q8 vector specialization in `fattn-vec.cuh`, and existing rocprof/patch-hit telemetry.

Before spending a production E2E lane on 1300:

1. On gfx1100 and gfx1201, profile the existing n=1 Q8_0/D=256 vector path at context depths 8K, 32K, 80K and >=160K where memory permits. Record vector-kernel time, achieved bandwidth, occupancy/VGPRs, and bytes implied by KV geometry.
2. Run a disposable compile-time sweep of only `nthreads_KQ_q` values supported by the current vector template (baseline 2 plus the next legal values). Do not retain a runtime knob unless at least two production shapes require different winners.
3. Require backend-reference/greedy parity for every candidate and unchanged work accounting. Reject any apparent speedup that reduces attended KV rows/heads.
4. Continue with 1300 selector qualification only if the best legal vector tuning improves n=1 Q8 attention kernel time by >=5% on either gfx1100 or gfx1201 and does not regress the other architecture by >2%, or if profiling proves the competing tile path pays >=3% of decode wall in avoidable Q8->F16 staging.
5. If neither condition holds, retire 1300 and close the dense-attention half of RNX02. Keep QSA/1301 separate as already specified.
6. If the gate passes, run same-binary ABBA at 8K/80K/160K with MTP disabled first, then MTP enabled only to verify that n=1 target decode and multi-query verify retain their independently selected kernels. Promotion requires >=1% E2E decode gain and no MTP acceptance regression.

### Consolidation

1298 and 1300 are not two permanent mechanisms: 1298 is rejected evidence for Q>2; 1300 is only a bounded selector experiment for n=1. Any useful `nthreads_KQ_q` result belongs in the existing upstream vector-kernel tuning seam, not in a new scheduler/cache/dispatch table. If the sweep wins, fold the architecture/shape choice into RNX02's existing FA ownership and retire the experiment knob after qualification.

### External evidence classification

- **Measured upstream evidence:** llama.cpp #27796 reports gfx1201 Qwen3.6-27B D=256 at d16384: F16 23.44 t/s, Q8_0 22.43 t/s (-4%), Q4_0 18.56 t/s (-21%). This is useful RDNA4 mechanism evidence, not a BigCherry benchmark.
- **Upstream hypothesis:** flat `nthreads_KQ_q = 2` may under-parallelize quantized KV dequantization. Treat only the sweep/profiler result as causal evidence.
- **Measured BigCherry evidence:** 1298 MTP3 vector override was ~6% slower at ~30K and is rejected.

## 2026-10-14 ownership boundary (QFP03 / BCOP202)

RNX02 retains D=256 quantized-KV attention selector/1300 ownership. The historical QSA/top-k proposal in RNX02 and 1294's `plan-ids=[RNX02]` do **not** authorize a second TOP_K implementation: QFP03 owns deterministic radix tie/partitioned ordering and its bounded 1294 dead-gather qualification. QFP15 owns residual draft acceptance nondeterminism; QFP04 consumes ordered indices. No RNX02 kernel or attention dispatch is changed by this audit.

## 2026-10-19 implementation audit — RDNA4 Q4_0 GQA block-per-lane vector FA (BCOP325)

**Disposition: upstream-admission candidate, NOT implemented or qualified in BigCherry.** This is a *new* 19 October kernel mechanism and gfx1201 hardware result in [llama.cpp #30803](https://github.com/ggml-org/llama.cpp/pull/30803/files), not a rerun of BCOP320's Q8 native TILE, the rejected 1298 multi-query vector override, or 1300's default-off Q8 selector. No local kernel fork, route knob, experiment queue or benchmark is authorized until Gate 1 below. Keep Q4_0 K+V / D256 / single-query decode as the first isolated target; do not generalize external results to Q8, D128 or MTP3.

### Exact pinned and upstream execution path

1. Pinned llama.cpp **b11552** (23b0202a189c44a54625aadcb37a946dd1d6278d), ggml/src/ggml-cuda/fattn.cu::ggml_cuda_get_best_fattn_kernel, already selects BEST_FATTN_KERNEL_VEC for quantized K/V with Q->ne[1] <= 2, supported D <= 256 (not 192), and K->ne[1] divisible by FATTN_KQ_STRIDE=256, unless earlier MMA/WMMA predicates win. For D256 the AMD WMMA gate requires Q->ne[1]*gqa_ratio_eff > 16. Confirm the *executed* route rather than assuming it from tensor dtype. ggml_cuda_flash_attn_ext_get_alloc_size and fattn-common.cuh::launch_fattn own the existing optional F16 staging, output scratch, stream, occupancy and graph lifetime.
2. Pinned fattn-vec.cuh::flash_attn_ext_vec and ggml_cuda_flash_attn_ext_vec_case use **one query head per z block** and RDNA nthreads_KQ_q=2 for quantized K. The Q4_0 K block is reloaded for each GQA head; a quantized vector specialization already exists. Patch 1300 only forces Q8_0/D256/Q=1 onto this old VEC kernel; it does not optimize Q4_0 or share GQA loads.
3. #30803 changes fattn-vec.cuh and fattn-common.cuh, NOT the GGML graph: on HIP RDNA4 with D=128/256 and Q4_0/Q8_0 K, KQ_blk assigns a whole 32-value K block to one lane (nthreads_KQ=D/32, i.e. 8 for D256). load_fattn_vec_K_blk loads 16 Q4 packed bytes + FP16 scale once; vec_dot_fattn_vec_KQ_blk performs two dp4a nibble products against Q8_1, with the 8*sum correction. For Q4_0/Q8_0 V at D256, V_wide handles 8 elements/lane. Existing static LDS softmax/warp reductions and the 128-thread launch remain; launch_fattn still requests zero *dynamic* shared bytes, not zero total LDS.
4. New ncols2=2/4/6/8 batches GQA heads within the same K/V head. The host predicate requires gfx120x, a mask, max_bias=0, logit_softcap=0, K length divisible by 256, supported Q4_0/Q8_0 K+V, and a divisible GQA ratio; otherwise ncols2=1/native fallback. For 24 Q / 4 KV heads (GQA6), the host selects 6 before 4 or 2. Kernel indexing maps z to (sequence, head0), reads K/V for head0/gqa, and writes every Q-head output and split-K meta under its original index. Sinks remain per Q head. Do not change the GQA/mask/softcap restrictions without a separate proof.
5. The upstream diff also changes fattn.cu's D128 quantized WMMA threshold (>8 to >16 effective Q columns) and adjusts V dequantization width/unrolling. **Do not attribute D128 or mixed-K/V gains to D256 Q4 GQA batching.** Source comments warn that fully unrolled KQ/V loops can increase VGPRs drastically and spill; the proposed code caps unroll at four for high ncols/mixed Q8-K/Q4-V. Inspect compiled gfx1201 ISA, VGPR, scratch, LDS and occupancy before promoting.

### Work/traffic, evidence and hardware applicability

- For Q4_0, each D256 K or V row is 8 blocks * 18 B = **144 logical bytes**. At GQA6 the original per-head VEC may issue up to 6*(144+144)=1728 logical K+V load bytes per KV token; ideal one-CTA reuse issues 288, a **1440 B/token/KV-head** logical opportunity. At 65,536 KV cells and four KV heads this is **360 MiB per attention layer** of potentially redundant *issued* reads, NOT measured HBM traffic or an E2E bound: caches, split-K, padding and actual memory transactions must be profiled. Every KV row, every query head, mask and output still must be processed.
- **External, not BigCherry:** #30803 R9700 gfx1201 / ROCm 10.1 / 300 W reports Qwen3.5-27B Q4_0 K+V, D256: tg64@65,536 **22.03 -> 26.76 t/s (+21.5%)**, @131,072 **17.81 -> 25.17 (+41.3%)**. Qwen3.5-35B-A3B Q4_0 K+V @65,536: **61.44 -> 77.10 (+25.5%)**. Its 4030/4030 FLASH_ATTN_EXT backend-op tests are upstream tests, not a BigCherry build. No confidence intervals or first-party gfx1201 receipt were supplied.
- **First-party historical control:** 1298's forced multi-query Q8 VEC at ~30K/MTP3 was slower (58.9 vs 56.1/54.3 ms/step); it had no GQA KV sharing. It does not disprove this new *single-query* block-per-lane path, but forbids extrapolation to MTP3. The production Flash-Next target's KV was previously placed on dual gfx1100 XTX ranks, not the gfx1201 R9700: a single-R9700 positive result is not automatically a three-rank Flash-Next benefit. gfx1030 draft and lack of normal GPU P2P are not candidates for this RDNA4 kernel.

### Cheapest discriminator completed (host only)

- The disposable oracle q4_gqa_vec_oracle.py tested **65,536** random packed Q4_0/Q8_1 blocks: exact signed integer dot after nibble compensation; **4,320** unique (sequence,Q-head,KV-head) mapping tuples; **15** route/negative controls; Python compilation passed. A scalar F32 expression with different association was bit-different in **41,211/65,536** samples (max absolute difference 0.00292969). This is a host rounding witness, **not** a GPU error, and requires reference/logit checks. No HIP build, backend-op test, rocprof or BigCherry benchmark ran.

### Bounded implementation and qualification gate

**Gate 1 — route/ceiling, read-only.** On an idle, explicitly assigned gfx1201, first identify a real BigCherry Q4_0 K+V / D256 / GQA2,4,6,8 model and physical rank (single-R9700 Qwen3.5-27B or 35B-A3B is an eligible control only if it fits). Record GGUF/KV type, Q/K/V ne/nb, n_seq, mask, max_bias, softcap, K padding, actual VEC vs TILE vs MMA selection, ncols2, graph UID, stream, physical GPU UUID, compiled architecture, executed kernel, K/V bytes, and target/MTP query widths. Use existing kernel-step-census/rocprof and 1300 marker where applicable, not a second telemetry system. If no gfx1201 KV route, terminal **NO_RDNA4_KV_ROUTE**. If measured *serialized* removable time is <2.9126% of baseline E2E time, terminal **BELOW_CEILING**. A sum of overlapping per-rank kernel times is not that ceiling.

**Gate 2 — pin/implementation.** Check #30803 merge state and approved llama.cpp pin before any code edit. Prefer upstream absorption, then isolate ncols2=1 versus ncols2=GQA in the same pinned binary/controlled build. If unmerged but Gate 1 passes, prototype only the Q4_0 D256 K/V load/reuse inside existing fattn-vec.cuh + fattn-common.cuh launch, using upstream exact guards and native fallback; do not add another FA dispatcher, scratch allocator, per-model config or persistent branch. Keep #30792's Q8 TILE mechanism and 1300's Q8 selector separately bisectable.

**Gate 3 — correctness and physical work.** Backend-op Q4_0/Q4_0 D256 with GQA2/4/6/8, q=1/2 and q=3 negative selector, KV lengths 256/512/1024 and 8K/32K/65K/131K, padded/tail rows, multiple sequences, sinks, nonzero ALiBi, softcap, one-sided quantized K/V and permuted strides. Compare ncols2=1 and grouped results to CPU F32 and native backend: exact head-to-KV assignment, output/meta writes, attended KV count, no missing/duplicate work, finite logits, full-vocab KLD/greedy, MTP acceptance, same-process multiple requests, multiple ubatches, long-context graph replay/reallocation and memory safety. Inspect ISA, VGPR/spills/LDS/wave32, achieved bandwidth and physical transactions. gfx1100 and gfx1030 must retain unchanged fallback; never use gfx1201 results as proof of gfx1100 improvement.

**Gate 4 — paired performance.** On the same R9700, paired baseline versus upstream Q4 GQA arm at 8K/32K/65K/131K where VRAM fits, q=1 and q=2, with MTP0 and MTP3 control; report per-step latency, pp512 unaffected control, actual K/V work/traffic, CI95 and allocation/graph changes. Require CI95-low **>=3% E2E throughput gain** with no material (>1%) control regression, no correctness/acceptance loss, and complete work. Otherwise terminal NO_GAIN / REGRESSION / NUMERIC_FAIL. Do not queue this alongside active RCS09/RAD11/RAD12/RAD10 GPU lanes.

### Ownership, upstream and other engines

RNX02 alone owns D256 quantized-KV VEC source/selector; 1300 remains an untested Q8-only control. BCOP320's recently audited native Q8 TILE admission (#30792) is **distinct** from this new Q4 VEC GQA reuse and must not be double-counted. QFP04/1295 owns sparse QSA decode, PRBE02/1266 owns WMMA sync, PRBE54/1271 Q5 F16 conversion, QFP07 placement, QFP15 acceptance. No changes to their active plans or implementation. vLLM's [ROCm Triton attention backend](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) and [KV dtype table](https://docs.vllm.ai/en/v0.25.1/design/attention_backends/) provide a grouped-query/paged-KV test comparator; its per-token-head INT4/INT8 layouts are not ggml Q4_0/Q8_0 and cannot be transplanted without independent AMD tests. **Novelty:** #30803 was opened 2026-10-19 09:42 UTC, after the earlier RNX02/BCOP320 audit; it supplies a new GQA-reuse kernel and direct gfx1201 Q4_0 measurements. Last repository-visible independent RNX02 work was outside the preceding 12 hours; no matching BigCherry PR, patch or queued receipt was found. Protected RCS09-04/05/06/07, RCS05, RAD11, RAD12, RAD10, WSL and release work remains untouched. Off-repository hardware queues are unverified.

## 2026-10-19 b11552 kernel audit — vector-FA KQ shared-memory WAR, BCOP313
**Disposition: upstream fence is non-HIP, with NVIDIA-only sanitizer evidence; no HIP kernel change admitted.** This is a new, narrow synchronization finding, not a repeat of the 1300 selector proposal. Upstream [llama.cpp #30573](https://github.com/ggml-org/llama.cpp/pull/30573), open at 2026-10-18 20:10 UTC, adds `ggml_cuda_syncwarp()` after the V-accumulation loop of `flash_attn_ext_vec`. The author reports 800,742 NVIDIA sm_86 compute-sanitizer WAR warnings before and zero after; 3,993/3,993 selected FA correctness cases passed after. **Those are NVIDIA measurements, not AMD race or speed evidence.**
### Exact source and AMD dispatch (pinned b11552, `23b0202a189c44a54625aadcb37a946dd1d6278`)
- `ggml/src/ggml-cuda/fattn-vec.cuh::flash_attn_ext_vec<D,ncols,type_K,type_V,use_logit_softcap>`: 128-thread CTA, 4 wave32 warps on RDNA. Quantized RDNA K uses `nthreads_KQ_q=2`; D256 quantized V uses `nthreads_V_q=32`. KQ scratch is `KQ[j*nthreads+tid]`, where `tid=WARP_SIZE*threadIdx.y+threadIdx.x`. The V loop reads `KQ[j*nthreads+k]`, where `k=threadIdx.y*WARP_SIZE+k0+(nthreads_V==WARP_SIZE?0:threadIdx.x/nthreads_V)`. For legal divisors of WARP_SIZE, each read stays inside the **same wave's** written 32-lane segment. This proves lane-index ownership, **not** synchronization/codegen safety.
- Existing `ggml_cuda_syncwarp()` immediately before the V loop is also relevant; #30573 adds the matching post-consumption fence before the next KV iteration can overwrite KQ. Neither changes KV tile count, attended cells, Q8 dequantization, K/V traffic, arithmetic, scratch allocation, CTA count, or graph topology. A hypothetical NVIDIA cost is one warp fence per KV iteration, not a bandwidth reduction.
- **Critical HIP difference:** pinned `ggml/src/ggml-cuda/common.cuh::ggml_cuda_syncwarp` wraps `__syncwarp()` inside `#ifndef GGML_USE_HIP`. Therefore both the existing and proposed calls compile to **no explicit HIP warp barrier**. Cherry-picking #30573 as written cannot add an AMD fence or an AMD speedup. Wave32 lockstep makes the NVIDIA independent-thread scheduling hazard non-transferable without actual gfx1100/gfx1201 ISA, compiler or sanitizer evidence. Do not claim the HIP path is proven race-free merely from this source fact.
- `ggml/src/ggml-cuda/fattn.cu::ggml_cuda_get_best_fattn_kernel`: for D256, Q8_0 K/V, single query, legal padded K (`K->ne[1] % FATTN_KQ_STRIDE == 0`), AMD MFMA/WMMA thresholds do not preempt the final quantized `Q->ne[1]<=2` vector branch. Native vector dispatch is already selected. 1300's opt-in early return omits the native KV-padding predicate and therefore can bypass its safety gate. **Retire the 1300 selector hypothesis; do not promote it.** Keep the package untested/inactive for historical reproducibility, not as an implementation dependency.
- `ggml_cuda_flash_attn_ext_vec_case_impl` uses the existing `launch_fattn` and backend graph/scratch ownership; no separate stream, allocator, cache, telemetry, or dispatch table is needed. Q>2 MTP verify and GQA remain on their independent stock paths. RNX02 owns D256 dense vector attention; QFP03/QFP17 own QSA/indexer, PRBE54 owns Q5 conversion, and 1298 is the rejected broad vector override.
### Completed cheap discriminator; remaining conditional gate
Host-only `warp_kq_ownership_oracle.py` checked **99 wave-local, complete-coverage cases and 216,576 modeled shared reads** for wave32/64, legal V lane divisors, 1/2 columns and 1/2/8 KV partitions; all passed. Source inspection verified the HIP no-op wrapper. These are **not** HIP tests or sanitizer results. No new GPU benchmark, kernel ISA, model logits, KLD or MTP run was performed. Existing BigCherry 1298 MTP3 vector override was ~6% slower (58.9 ms/step vs 56.1/54.3 ms controls at ~30K), but that is not a KQ-fence A/B.
**Terminal by default:** `UPSTREAM_NON_HIP_ONLY` / `NATIVE_VEC_ALREADY_SELECTED`; no patch, new flag or GPU queue. Reopen only on (a) a reproducible gfx1100/gfx1201 KQ WAR or miscompile receipt tied to this kernel and exact compiler/pin, or (b) a current-pin n=1 Q8_0/D256 kernel-critical-path receipt proving an independent >=3% E2E removable ceiling. For (a), first inspect wave32 ISA and compare stock vs a disposable **HIP-valid** wave-level fence (not `ggml_cuda_syncwarp`), preserving all-wave participation; test full KQ read-before-write, short/long KV tails, multi-request/ubatch, graph replay, logits/KLD/greedy and MTP. For (b), use the existing RNX02 `nthreads_KQ_q` sweep and rocprof lane, not the redundant 1300 override. A/B requires paired same-pin gfx1100/gfx1201 runs, unchanged physical work/bytes, CI95-low >=3% E2E and <=1% controls; otherwise reject. gfx1030 is a negative control, not an assumed beneficiary.
**External comparison:** vLLM ROCm's paged-attention backend and AITER attention kernels use different cache layouts, formats and synchronization ownership; they are references for AMD qualification methodology, not a drop-in GGUF Q8_0 `flash_attn_ext_vec` barrier. See [vLLM ROCm attention](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py) and [ROCm AITER](https://github.com/ROCm/aiter).
