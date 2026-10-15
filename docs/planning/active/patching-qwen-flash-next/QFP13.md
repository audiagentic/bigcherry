---
id: QFP13
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T17:26:47.771591+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Decode kernel-count reduction for Flash-Next (launch-gap bound: ~3.5 us gap per ~1.7 us kernel)

## 2026-10-14 b11552 rollback-copy route reconciliation (BCOP209)

**Disposition:** Retain validated 1308 and its default-on direct-strided snapshot copy. The historical `~108 copyBuffer*/token` item is **not unassigned**. The old v2 census attributed ~89 to recurrent rollback snapshot `CONT + CPY` pairs and ~19 to other copy neighbourhoods. 1308 removed the intermediate CONT, **not the required cache snapshot CPY**. A current-pin residual runtime-copy count cannot be obtained by subtracting ~44 from ~108: the surviving direct-strided CPY can dispatch as a *different kernel class*. Historical 1270 -> 1212 kernels/token and +3%/+3.6% adoption measurements are first-party, pin/profile-bound, not new b11552 measurements.

### Actual implementation and lifetime

- `src/models/qwen4exp.cpp::graph::build_conv_state_at` constructs `conv_input = CONCAT(state, TRANSPOSE(x))`, then for each of `n_rs_seq+1` rollback slots builds a strided `VIEW_3D tail` and a cache `VIEW_2D dst`. The original `ggml_cpy(ggml_cont(tail),dst)` materializes an intermediate; 1308 replaces it with `ggml_cpy(tail,dst)` (default on; `BIGCHERRY_ROLLBACK_NO_CONT=0` restores the original).
- `ggml/src/ggml-cuda/cpy.cu::ggml_cuda_cpy` chooses contiguous `cudaMemcpyAsync`, then `ggml_cuda_cpy_as_memcpy_2d`, then type-specific kernels. The 2D fast path requires `ggml_are_same_shape(src,dst)`. Production multi-channel rollback source `[state_cols,channels,n_seqs,1]` and flattened cache destination `[state_cols*channels,n_seqs,1,1]` differ in shape, so that fast path is **not** admitted; for F32-to-F32 the direct-strided route reaches `ggml_cpy_scalar_cuda`. Do not describe it as an equivalent one-shot DMA copy or assume its kernel appears under the old `copyBuffer*` name.
- `ggml/src/ggml-backend-meta.cpp` handles `GGML_OP_CPY` through `handle_cpy` and `CONT/RESHAPE` through `handle_reshape`; the eliminated graph node changes split/allocation topology. The cache destination must be fully written for every slot/sequence before any rollback reader; input-view backing and destination lifetimes must survive graph reuse and multi-request reuse.
- Existing upstream `ggml_cuda_try_gdn_cache_fusion` and `ggml_cuda_try_ssm_scan_cache_fusion` fold **other** state-tail writes when their exact producer/view/contiguity/shape guards hold. They do not match Qwen4Exp's `CONCAT`-owned convolution tail. Do not add a duplicate generic state-cache fusion matcher.

### Narrow residual-copy gate, not another implementation plan

1. Re-census the **current b11552 composed production profile** with 1308 on/off, same prompt and MTP depth; report per generated token and per rank separately: runtime `copyBuffer*`, `cpy_scalar`, `CONT`, `CPY`, their launch IDs and device durations. Associate every candidate with graph node, producer/view ancestry, `ne/nb`, slot/sequence, owning rank and stream via correlated HIP activity. Do not equate kernel-name counts with GGML node counts or infer overlap from adjacent timestamps.
2. First isolate the historical ~19 **nonrollback** copy neighbourhoods. Classify each as required state write, input staging, scalar broadcast, or removable intermediate. Existing QFP16/1326/QFP27 owns scheduler input lifetimes; 1308 owns rollback; QFP13 only ranks measured residuals and routes a qualified candidate to its existing owner.
3. Cheap controls before GPU work: compare direct-strided vs CONT+CPY on host for slot 0..3, `n_seqs=1..3`, channels and tail widths; reject any shape/stride mismatch. Host-only fixture passed **216 synthetic semantic cases, 2 shape-admission guards and 9 launch-count identities**; Python compilation passed. This is **not** a HIP/Meta correctness test.
4. A deliberately optimistic arithmetic illustration for the *19 historical nonrollback launches* is `19*(3.1+1.6)..19*(3.7+1.8) us = 0.0893..0.1045 ms/token` if every global-median gap and kernel duration is serial and removable. These are **not measured copy costs or rigorous upper bounds**; actual launch duration, overlap and route must be measured. Do not schedule a new fusion unless the correlated critical-path saving can clear QFP13's existing `>=0.10 ms/token` or `>=0.5% E2E` acceptance gate.
5. Qualify any future owner-local change with same-process sequential requests, multiple sequences, ubatches, MTP depths `0/3/7`, cache rollback/replay, long context, graph reallocation, exact greedy/logit and draft acceptance comparisons, and rank-specific GPU tracing. No new scheduler, allocator, generic copy fusion or trace framework.

**Terminal:** `ALREADY_COVERED_1308` for the rollback pair; `UNVERIFIED_ROUTE` if kernel-to-node correlation fails; `REQUIRED_STATE_WRITE` for necessary cache copies; `BELOW_EXPOSED_GATE` if the measured residual cannot meet the QFP13 gate. Otherwise route a *single* bounded correction to the existing owner. vLLM's `causal_conv1d_update` writes indexed state in-kernel, but its packed state and scheduler semantics differ; it is a correctness/mechanism reference, not an AMD performance claim. Upstream llama.cpp #25057's 2D memcpy path already exists in b11552 and is excluded here by shape.


## Description

QFP13 is the Flash-Next **measurement, ranking, and acceptance umbrella** for local decode launch reduction. It does not own backend fusion implementations.

Production profile-v2 decode (`flashnext-v2-profile/d8192`, rocprofv3) shows a median same-stream inter-kernel gap of ~3.1 us on XTX and ~3.7 us on R9700, versus ~1.6-1.8 us median kernel duration. Decode executes roughly 1,300 kernels/token per tensor-split GPU. The immediate lever is therefore fewer recurring graph nodes/launches, not another graph-cache layer.

QFP13 owns:
- the adjacent-kernel/n-gram census and per-token opportunity counts;
- ranking candidates by **measured removable launches and serial time**, not source-code intuition;
- assigning each proven topology to one canonical implementation owner;
- the common end-to-end launch/TG acceptance gate;
- re-profiling after every accepted optimization and advancing to the next ranked candidate.

Implementation details belong in the canonical owner item. Do not duplicate matcher/kernel designs here.

## Steps

1. Maintain an adjacent-kernel/n-gram census for the selected production decode window on gfx1100 and gfx1201.
2. Re-census the current-pin residual nonrollback copy neighbourhoods using node-to-HIP-event correlation. The old ~108/token `copyBuffer*` baseline was already largely attributed to 1308; do not reopen it as unassigned.
3. For each ranked candidate, record: frequency/token, maximum removable launches/token, expected serial gap time, semantic risk, and canonical owner.
4. Send implementation work to that owner (PRBE38/39/40/05/etc.); QFP13 must not gain backend matcher/kernel pseudocode.
5. Re-run the same census after each owner lands an accepted experiment. Measure actual launch reduction, gap-time reduction, GPU busy share, ms/step, and effective TG.
6. If an experiment misses the gate, park/supersede that implementation path rather than broadening it speculatively; advance to the next measured candidate.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Upstream b11126 reference used for ownership: canonical supported `GGML_OP_GLU` vector graphs already feed `ggml_cuda_mm_fusion_args_host/_device` into fused `mul_mat_vec_q/f` epilogues. Literal `UNARY(SILU|SIGMOID|SOFTPLUS) -> MUL` is instead handled through the separate `ggml_cuda_op_unary_mul` pointwise path, so PRBE38 remains a distinct candidate only where trace proves that pointwise work remains a separate post-GEMV launch/global-memory round trip.

QFP11's boundary decomposition measured roughly ~25 us for the split/round-trip itself while arrival skew was much larger; QFP06 measured ~2 MiB per cached graph instance and severe recapture churn with a cap of 32 against a ~195-instance live working set. Those are supporting evidence, not new QFP13 implementation seams.

Related: PRBE05, PRBE06, PRBE37, PRBE38, PRBE39, PRBE40, QFP06, QFP09, QFP11, RNX04, RNX08, RNX10, RNX11.

2026-10-04 1307 (PRBE05 stage 2, Q8_1 activation reuse; 1235 re-anchored on upstream common.cuh so it builds outside tuning sources): quantize_q8_1 per generated token per XTX 183 -> 149 (-19%), kernels/token ~1307 -> 1270; profile v2 quick screens ~24K 46.9 vs 48.0/46.6 ms/step, ~80K 53.1 vs 55.0/53.7 - neutral to ~1-2% (first baseline arm runs high every screen, so treat as drift-limited). Acceptance identical (exact reuse). Only ~1/5 of quantizes share an input; the remaining ~150/token need fusion (rms_norm->quantize 41, unary/unary_gated->quantize ~60) or no-quantize F32-activation matvec (1241/1274 route) to remove.

2026-10-04 1308 (rollback snapshots without CONT, BIGCHERRY_ROLLBACK_NO_CONT=1; source: qwen4exp.cpp [TAG_RECURRENT_ROLLBACK_SPLITS], n_rs_seq+1 = 4 slots per recurrent layer, each CONT + CPY = 8 runtime copy kernels per layer, ~89 of ~108 copies/token): with 1307 on in both arms, greedy output IDENTICAL; kernels/token 1270 -> 1212 (cumulative with 1307: 1307 -> 1212, -7%); quick screens ~24K 46.0 vs 46.8/46.1 ms/step (neutral), ~80K 52.2 vs 54.3/53.3 (~-3%, outside both baselines). Next: full ABBA profile v2 vs v2+1307+1308 at 10K/80K for adoption.

2026-10-04 ownership-map update for the 'Runtime copyBuffer* pair (~108 kernels/token) - unassigned evidence task' row: ATTRIBUTED and IMPLEMENTED. Kernel-neighbourhood attribution on flashnext-v2-profile/d8192 (no API trace needed): ~89 of ~108 copies/token are runs of 8 copies after concat_non_cont, from qwen4exp.cpp [TAG_RECURRENT_ROLLBACK_SPLITS] (n_rs_seq+1 = 4 rollback slots per recurrent layer, each ggml_cpy(ggml_cont(tail), dst) = 2 copies). Owner: patch 1308_qwen4exp_rollback_copy_no_cont (BIGCHERRY_ROLLBACK_NO_CONT=1, copies the strided tail directly). Evidence: greedy IDENTICAL; kernels/token 1270 -> 1212 on top of 1307; ~80K quick screen ~-3% ms/step. Adoption ABBA (v2 vs v2+1307+1308, flashnext-v2-fusion-ab-2): ~10K decode 73.8/73.6 -> 76.0/75.7 t/s (+3%, complete separation); ~80K pending. Remaining ~19 copies/token: singleton copies before k_bin_bcast (unary/get_rows/cpy_scalar neighbours) - unassigned.

2026-10-04 Q8_1 producer stack result (flashnext-v2-q81c, profile v2 + 1307 cache + 1308, new arm adds 1309 BIGCHERRY_RMS_Q81 + 1310 BIGCHERRY_ACT_Q81; 1307 now also looks up a contiguous padding-free RESHAPE under its view_src's key): greedy IDENTICAL; quantize_q8_1 per token per XTX 183 (v2) -> 149 (1307) -> 104 (+1310 first cut) -> 74 (+reshape lookup, 1309 hits); kernels/token 1307 -> 1138 (-13%). Quick screens vs 1307+1308: ~24K 46.0 vs 46.1/45.9 ms/step (neutral), ~80K 51.6 vs 53.5/52.9 (~-2.5..-3.5%). Diagnosis tooling: BIGCHERRY_Q81_TRACE publish/miss pairing (tools/lab/flash-next/q81-trace-run.sh) found the hc_norm RESHAPE-of-RMSNorm keying and per-GPU activation width 320 (why 1310 needed non-512 rows). Rejected: 1310 row cap 512 (publishes ~3x more routed-expert activations that MoE MMVQ never hits; ~24K +6% regression) - reverted to 16. Remaining misses per trace: DSV4_HC_PRE outputs ~2958, other RESHAPE ~2896, GLU (routed experts) ~1847, MUL ~256. Next: producer for dsv4_hc_pre output; MoE (ids) MMVQ key alignment, then re-raise the row cap.

2026-10-04 1311 (BIGCHERRY_HC_Q81=1, hyper-connection pre-mix emits Q8_1) on top of 1307-1310 with 1310 row cap 64 (flashnext-v2-1311): greedy IDENTICAL; quantize_q8_1/token/GPU 74 -> 45 (cumulative from profile v2: 183 -> 45, -75%); kernels/token 1138 -> 1116 (cumulative 1307 -> 1116, -15%); screens ~24K 45.4 vs 47.0/45.5 ms/step (neutral), ~80K 51.1 vs 52.8/52.2 (~-2%). Trace: DSV4_HC_PRE misses 2958 -> 1246, RESHAPE 2896 -> 936; GLU misses unchanged (1557) and publish-act unchanged even with cap 64 -> the routed-expert ffn_moe_swiglu outputs are not produced by ggml_cuda_op_unary_gated (likely the fused MUL_MAT_ID+GLU MMVQ path writes them), so 1310 cannot publish them; catching them needs a Q8_1 write in the fused GEMV epilogue (PRBE37 scope). Planned generalisation (owner request): replace the producers' row caps with the MMVQ batch rule (publish only when the graph's token count <= MMVQ_MAX_BATCH_SIZE) so other models' fan-out shapes and small prefill ubatches behave correctly.

2026-10-04 1312 (fused UNARY*MUL -> Q8_1 via 1310; upstream fuses sigmoid+mul into ggml_cuda_op_unary_mul so ggml_cuda_op_mul never runs for these) build A/B v3 vs v3+1312 (flashnext-v3-1312b): greedy IDENTICAL; ~24K 45.3 vs 47.7/45.2 ms/step, ~80K 51.0 vs 52.5/51.1 (neutral); quantize/token/XTX 45 -> 41.3, kernels/token 1116 unchanged. Trace: attn_gated misses gone (publish-act attn_gated); GDN final_output still missed - the MUL node is [128, heads, T] and ssm_out reads reshape_3d(128*heads, T) whose 1536-2688 row needs MMVQ padding, so 1307's padding-free reshape lookup cannot match. Fix committed: 1310 helper flatten01 (+ gate index by source row stride), 1312 publishes the GDN gated norm flattened, 1307 flattened-reshape lookup; recheck queued (flashnext-v3-1312c).

2026-10-04 1313 (BIGCHERRY_SCALE_ACT_FUSE=1: SCALE -> SILU/SIGMOID [-> SCALE] in one launch, hyper-connection blocks; bit-identical math, 1310-style Q8_1 when the chain ends at the activation) env screen on one build (flashnext-v3-1313b): greedy IDENTICAL; ~24K 43.7 vs 45.2/44.5 ms/step (-2..-3%), ~80K 50.1 vs 51.1/51.1 (-2%); t/s flat at 80K because drafted/accepted counts differ (169/258 vs 171/252, 170/255) - run-to-run draft nondeterminism (QFP15), also seen between the two baseline arms. Census: kernels/token 1116 -> 1024 per XTX (972 R9700), elementwise ~357 -> ~275/token. Cumulative from profile v2: 1307 -> 1024 kernels/token (-22%). Candidate for v3 adoption after a multi-request ABBA (single-request t/s is acceptance-noise-limited until QFP15 is fixed).

2026-10-04 1314 (BIGCHERRY_AR_FUSED=1: one fused produce+consume kernel per small cpu-root AllReduce) env screen (flashnext-v3-1314b): greedy IDENTICAL; ~24K 44.3 vs 44.3/44.2 ms/step, ~80K 51.1 vs 51.1/51.1 - neutral. Census: AllReduce launches 59 -> 29.5/token, kernels/token 1116 -> 1072, but AllReduce busy time 1.2 -> 2.5 ms/token on XTX: the wait for the CPU sum moved from the inter-kernel gap into the fused kernel's spin. The AllReduce is bound by rank arrival skew and the CPU sum, not by the launch. PARKED (correct, neutral); do not adopt.

2026-10-04 1312 recheck after the GDN flatten fix (flashnext-v3-1312c, build A/B v3 vs v3+1312): greedy IDENTICAL; ~24K 44.3 vs 44.8/44.9 ms/step, ~80K 51.0 vs 51.8/51.1 (0..-1%); quantize/token/XTX 45 -> 29.8 (cumulative from profile v2: 183 -> 30), kernels/token 1116 -> 1091 (R9700 1064 -> 1040). Adopt with 1313 (v3 + 1312 + 1313 expected ~1000 kernels/token) after a multi-request ABBA.

2026-10-04 post-bump (pin 0504396) adjacent-pair census on v5+1327 (flashnext-census-1327, XTX0, per 1k kernels ~= per token): MMVF -> unary_op -> k_bin_bcast chains ~26/token (literal UNARY->binary after an f16 matvec; PRBE38 topology; fold into the MMVF epilogue, ~52 removable launches, ~1.5%); k_bin_bcast -> k_bin_bcast ~37/token (binary-op chains incl. the remaining #29819 remap arithmetic; fused binary-chain kernel, ~1%); cpy_scalar -> cpy_scalar ~34/token (runs of recurrent-state rollback copies after 1308; one batched copy kernel per layer, ~1%); quantize_q8_1 <-> mul_mat_vec_q_moe ~15/token (routed-expert path not covered by the Q8_1 cache); bc_cpu_root produce -> consume ~29/token (1314 parked). Kernels/token: old pin ~1000, new pin 1053, + 1327 1027. Next: identify the exact graph ops of the MMVF->UNARY->BINARY chain (likely GDN alpha/beta gating) and implement the epilogue fusion first.

## Current Evidence

Profile-v2 observations:
- elementwise: ~370-400 kernels/token;
- quantize: ~180-200/token;
- MMVQ: ~150-166/token;
- norm/rope: ~80-95/token;
- get/set_rows/copy: ~55-65/token by coarse class;
- QSA/top-k: ~60/token;
- AllReduce produce/consume: ~60/token.

2026-10-04 XTX0 ~10K census:
- `quantize_q8_1`: 183/token;
- `mul_mat_vec_q`: 154/token;
- `unary_op`: 102/token;
- `k_bin_bcast`: 98/token;
- `scale_f32`: 98/token;
- `mul_mat_vec_f`: 82/token;
- `rms_norm`: 79/token;
- HIP runtime copy kernels: ~108/token (`copyBufferRectAligned` + `copyBuffer`) in the historical pre-1308 census; ~89 attributed to rollback CONT+CPY, not an unassigned current-pin count;
- `unary_gated`: ~30/token;
- cpu-root AllReduce produce/consume: ~30/token each.

Important interpretation: an adjacency count is an **opportunity count**, not proof that the upstream value is reusable or that all launches can be removed. In particular, ~154 `quantize_q8_1 -> MMVQ` adjacencies/token do **not** imply 154 Q8_1 cache hits. PRBE05 must prove repeated consumers of the same activation identity inside a valid cache generation before claiming launch removal.

## Canonical Ownership Map

| Candidate / evidence | Canonical owner | QFP13 disposition |
| --- | --- | --- |
| Literal vector `MUL_MAT[/ID] -> UNARY(SILU/SIGMOID/SOFTPLUS) -> MUL` where the pointwise gate remains a separate launch | **PRBE38** | Rank by trace frequency; PRBE38 owns matcher, alias/broadcast proof, epilogue wiring, tests, and fallback. |
| Canonical `GGML_OP_GLU` / SWIGLU-family GEMV epilogue fusion and any distinct SIGMOID/GEMM capability gap | **PRBE37** | Do not duplicate in PRBE38/QFP13. b11126 already fuses the supported canonical GLU vector path. |
| GEMV/MMVQ -> VIEW -> residual ADD | **PRBE39** | Existing implementation owner. |
| Paired K/V projection / LoRA projection fusion | **PRBE40** | Existing implementation owner. |
| Q8_1 activation reuse/cache (`quantize_q8_1 -> MMVQ`) | **PRBE05** | Patch 1235 is historical/inconclusive; 1307 is the active A/B child experiment. Acceptance requires actual same-activation reuse and hit/miss evidence. |
| CUDA/HIP graph working-set memory / graph-cap evidence | **QFP06** | Negative experiment/evidence only: caps below the live split-graph working set cause recapture churn. Do not create another graph-cap item. |
| AllReduce split/graph-boundary removal | **QFP11** | Lower priority: measured split/round-trip boundary is a minority of the loss; rank-arrival skew dominates. |
| MMVQ -> SCALE / MMVF -> SCALE (~30 + ~30 adjacencies/token in one census) | **unassigned candidate** | Source + numerical proof first. Do not create an implementation item until exact topology, output dtype, rounding equivalence, and removable-launch count are established. |
| RMSNorm -> Q8_1 quantize | **PRBE06** | Existing quantization/fusion owner; keep separate from PRBE05 cache identity work. |
| Runtime rollback `CONT + CPY` pair (~89 of ~108 historical copyBuffer* launches/token) | **1308 / QFP13 (validated)** | Already implemented; preserve direct-strided cache writes, distinguish scalar CPY from runtime copyBuffer class. Residual ~19 nonrollback neighbourhoods are a QFP13 *measurement-only* gate, with implementation routed to an existing owner. |

## Promotion Rule For New Candidates

A QFP13 candidate may be promoted to an implementation owner only when all are true:
1. production trace proves the recurring adjacency/topology and frequency per generated token;
2. graph/source mapping proves the exact producer/consumer relationship and whether eliminating a node is legal;
3. current upstream source is checked so the optimization is not already present under a different graph representation;
4. numerical/alias/broadcast/dtype constraints are explicit and fail closed;
5. predicted removable launch time is large enough to meet the QFP13 acceptance gate.

If an existing plan already owns the seam, update/cross-link that plan rather than creating another implementation plan.

## Validation / Acceptance Gate

Use profile-v2 ABBA on the existing XTX+XTX+R9700 tensor-split topology at shallow (~8-10K) and deep (~65-80K) context.

For every promoted optimization require:
- exact target adjacency/kernel count before and after;
- total kernels/token and relevant class count/token;
- inter-kernel gap ms/token and GPU busy share;
- target verify ms/step and effective TG;
- deterministic/greedy correctness against the unfused control;
- no compensating new launch or material occupancy/register regression;
- supported and negative/fallback cases covered by the implementation owner's tests.

Performance gate: prefer candidates with a credible >=0.5% end-to-end improvement; an accepted implementation must show either >=0.5% end-to-end TG/ms-step improvement or a measured >=0.10 ms/token serial launch-gap reduction without an end-to-end regression. Otherwise park it and re-rank.

## Scope Boundaries

Out of scope for QFP13 implementation ownership:
- graph-cache-cap work (QFP06 evidence; cap-below-working-set already disproven);
- AllReduce graph/scheduler redesign (QFP11/QFP09/RNX11); 
- backend matcher/kernel implementation already owned by PRBE items;
- speculative fusion based only on source adjacency or operation names;
- model/tensor-split/MTP-depth changes.

## Change Log

- 2026-10-03T17:26:47.771591+00:00 (created-by): Created by agent
- 2026-10-03T17:27:37.409987+00:00 (updated-by): Updated: section:notes
- 2026-10-03T19:38:00+00:00 (agent): Corrected branch selection to `patch-refactor`; made PRBE38 literal GEMV->UNARY->MUL the first trace-gated implementation target; added code-reuse, LOC-reduction, validation and performance gates.
- 2026-10-03T20:00:05.640046+00:00 (updated-by): Updated: section:notes
- 2026-10-04 (agent): Consolidated QFP13 as the profiling/ranking/acceptance umbrella; removed duplicate PRBE38 backend design; assigned canonical owners; separated opportunity counts from proven reusable launches.
- 2026-10-03T20:26:08.405034+00:00 (updated-by): Updated: section:notes
- 2026-10-03T20:32:18.645038+00:00 (updated-by): Updated: section:notes
- 2026-10-03T22:17:40.734913+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261003_221744_flash-next-decode-issues-13_8662
- 2026-10-03T22:17:47.230319+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-03T22:54:32.356647+00:00 (updated-by): Updated: section:notes
- chg_20261003_235210_flash-next-decode-launches-15_2384
- 2026-10-03T23:52:13.609111+00:00 (updated-by): Updated: section:ledger-events
- chg_20261004_011520_three-more-flash-next-decode-k_5440
- 2026-10-04T01:15:24.244427+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-04T04:00:41.792844+00:00 (updated-by): Updated: section:notes
- 2026-10-04T04:18:25.310777+00:00 (updated-by): Updated: section:notes
- 2026-10-04T04:47:25.709859+00:00 (updated-by): Updated: section:notes
- 2026-10-04T11:26:49.575199+00:00 (updated-by): Updated: section:notes
