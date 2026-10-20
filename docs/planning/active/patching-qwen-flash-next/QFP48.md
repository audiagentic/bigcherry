---
id: QFP48
order: 48
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-10T13:53:50.737727+00:00'
breadth: ''
skill: advanced
created-by: claude
priority: P2
work: M
---

# Fusion round 2: choose kernel fusions from the 2026-10-10 prefill trace and a decode trace

## Description

Owner, 2026-10-10: "can we fuse more kernels". The earlier fusion candidates came from an external report (PRBE11 / 37 / 38 / 39 / 40: GEMV epilogue, residual-add, K+V projection; parked in QFP35, now closed). This item picks fusions from our own trace instead.

What the trace says (run dp1, Flash-Next production profile on main at 1042e84d, rocprofv3 kernel trace, share of a target card's kernel time, the same at 24K and 98K): AllReduce 39-46% (QFP49), MMQ incl. MoE 25-30% (mul_mat_q 19.4 s, quantize_mmq_q8_1 1.37 s at 24K), flash attention about 10%, norm / activation / elementwise 4.6% (rms_norm_f32 1.10 s, k_bin_bcast 0.82 s, dsv4_hc_pre_grid 0.63 s), set / get rows and mask build 3.8% (k_get_rows_float 0.82 s, k_get_rows_float_vec 0.74 s), float matmul 3.8%, GDN 2.9%, other 6.1% (bc_moe_router_splitk_partial 1.93 s, bc_dsv4_hc_post_gate_grid 1.15 s, bc_mm_ids_helper_mw 0.82 s).

Fusion pays where two memory-bound kernels read or write the same data back to back. Candidates to check, each against the trace order of kernels (not assumed): quantize_mmq_q8_1 into its producer (rms_norm or the HC pre-grid), the two get_rows passes, k_bin_bcast into its neighbour, and the (now retired) router partial/reduce candidate; 1.93 s was the **partial GEMM compute**, not removable reduction time.

## 2026-10-17 single-branch reconciliation (BCOP239/BCOP275)

**Disposition: reject router partial/reduce adjacency as a fusion task without new evidence.** Validated 1357 uses independently scheduled split-K partial workgroups, backend-pool scratch and a fixed-order second reduction launch; a workgroup barrier cannot publish across partial CTAs. The October 10 measured +3.2% pooled Flash-Next prefill compares **1357 router replacement versus native SGEMM**, not partial/reduce fusion; changed greedy text and pooled decode 69.8 -> 67.2 tok/s forbid calling it bit-identical or a decode gain. Its README's obsolete “Not promoted” line has been corrected on this branch.

Leave QFP48 pending only for its other already-owned trace/batch-boundary work. Reopen reduction fusion **only** if an exact-pin warm per-rank profile demonstrates the *reduce kernel plus its launch gap* has >=3% exposed end-to-end removable cost, followed by a globally fenced completion/epoch plan that survives graph replay, cancellation, pool lifetime and fixed-order F32/top-k numerical controls. Otherwise terminal `ROUTER_REDUCTION_FUSION_NOT_ADMITTED`. QFP36/1357 owns the router implementation; no new patch, cache, dispatcher, scheduler or hardware queue is authorized here. The full older audit remains on the historical `docs/bcop239-router-reduction-published` branch; this is the trimmed decision.

## Steps

1. Add a decode trace mode to long-ctx-profile.sh (prefillprof exists; decode has none) and take it on the current build.
2. From both traces list adjacent kernel pairs by total time (same stream, consecutive, same tensor), per card.
3. For each pair above 1% of prompt wall or of decode step time: check existing fusions (1307-1313, 1344, 1355, native GLU) do not already cover it; write the candidate with its expected saving.
4. One patch per fusion, default off, lightweight tier: mechanics tests, marker, ABBA with complete separation, greedy identity.
5. Read the ExLlamaV3 RDNA3 ports (wtogami/exllamav3-rocm, phoenixhaxor/exllamav3-rocm-moe: "fused-core prefill kernels") for fusions we have not considered; reference only.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per fusion: offline mechanics + patch-lint, BIGCHERRY_PATCH_HIT marker or census showing the launch removed, ABBA on one binary with complete separation at 8K / 24K / 98K, greedy identity (GGML_CUDA_DISABLE_FUSION=1 control on both arms where the fusion pass is involved), no q4 KV.

## Effort & Risk



## Standards



## Acceptance Criteria

- A decode trace exists for the current build, with the same family table as the prefill one.
- Each fusion candidate from the trace is either promoted (lightweight tier) or rejected with its measurement.
- No candidate is taken from an external timing without our own trace showing the pair is adjacent and material.

## Notes

Depends on nothing; the AllReduce work (QFP49) is larger and comes first for prefill. Bandwidth per kernel (weight bytes read over kernel time, against 950 GB/s for a 7900 XTX and 640 for the R9700) is to be added to the trace table so a fusion's ceiling is known before it is written. Related: QFP44 (MMQ tile geometry), QFP46 (prefill concurrency contracts).

2026-10-11 census for the kernel-count lever (tools/lab/flash-next/kernel-step-census.py; a step = the kernels a card launches between two cross-card sums). Why it matters now: the standalone probe shows staggering hides 84% of a sum at 2 kernels a step but 42% at 40 and 13% at 80 (QFP49), and every launch leaves a gap. Real Flash-Next through the llama library, -ub 256, no drafter (trace lh2 seq), per RX 7900 XTX: 1,527 steps, 44.1 kernels a step (median 47, max 94); a step runs 983 us of kernels in 1,592 us - only 62% of a step is kernel time. Production profile -ub 512 (trace pipe4): 44.3 kernels a step (median 37), 1,746 us of kernels in 2,786 us (63%); R9700 37.4 a step. By kernel (ub256 trace): mul_mat_q 6.60 launches a step, 77.9 us each, 52.3% of kernel time, and a gap of 39.7 us after each one (other kernels leave about 4 us; k_bin_bcast 17 us) - 6.6 x 40 us = about 260 us a step, 16% of the step, 0.40 s of a 3.84 s prefill; quantize_mmq_q8_1 6.60 a step at 5.6 us; rms_norm_f32 2.66; k_bin_bcast 3.26; mul_mat_vec_f 1.50; bc_mm_ids_helper_mw 1.48; router split-K partial 0.49; flash_attn_ext_f16 0.13 a step at 305 us. Commonest adjacent pairs a step: quantize_mmq_q8_1 -> mul_mat_q 5.12; bc_mm_ids_helper_mw -> quantize_mmq_q8_1 1.48; quantize_mmq_q8_1 -> mmq_build_moe_block_map 1.48 -> mul_mat_q 1.48; rms_norm_f32 -> quantize_mmq_q8_1 1.38; mul_mat_q -> unary_gated_op_kernel 1.36; mul_mat_q -> stream_k_fixup -> bc_scale_act_kernel -> quantize 0.99 each. Two candidate levers, both without staggering's half-size tax: (1) the 40 us gap after every mul_mat_q - find what the host or the card does there (not yet known: host work in the MMQ launch path, a wait, or kernel-switch latency); (2) fewer launches around MMQ: quantize + mul_mat_q is 5 of the 44 launches a step as a pair, and the MoE chain ids-helper -> quantize -> block map -> mul_mat_q is four launches 1.5 times a step. Next: attribute the 40 us (host perf call graph around ggml_cuda_mul_mat_q, and a standalone launch-sequence probe), before any fusion code.

2026-10-11, attributing the gap after mul_mat_q (kernel-step-census.py now lists the gap per adjacent pair). It is not 40 us after every mul_mat_q but one transition: on the RX 7900 XTXs mul_mat_q -> convert_unary_cont_vec4 (the sum's bf16 conversion) 0.50 times a step with a 478 / 402 us gap (240 / 202 us a step): the card has finished its subgraph and waits for the host to submit the other devices before the sum is issued. On the R9700 the gap is after the sum: convert_unary_cont_vec4 -> next kernel, 781 us (0.25 a step) and 178 us (0.48 a step), about 280 us of a 1,713 us step: the host submits devices in order and reaches the R9700 last. Host profile of the same prefill (perf, slim harness, 16K tokens): 93% of host samples are in libhsa-runtime's wait loop, about 7% real work - the host is not short of time. TEST of the obvious reading (serve the slowest card first): -dev ROCm2,ROCm0,ROCm1 -ts 0.42,0.31,0.27 against the default order, slim harness, 16,384 tokens, -ub 512, six runs each (lo1-lo4): default 1,587-1,613 t/s, R9700 first 1,566-1,587 t/s. About 1% slower. So the R9700's post-sum wait does not set the pace; reordering only moves the gap. Open: what does set the step's length - the three cards each show about 38% of a step without a kernel running, in different places. Candidates not yet tested: the sum itself (cards arrive 1.3 ms apart), RCCL's own latency per sum, kernel launch latency inside the runtime. Do not build fusion or dispatch changes on the gap figures alone; each needs its own A/B like this one.

2026-10-11, which steps and which matmuls (tools/lab/flash-next/mmq-variant-census.py on the production trace pipe4: Flash-Next, -ub 512, MTP drafter; the window includes the decode that follows the prefill). Per RX 7900 XTX, steps classified by content: recurrent (GDN / SSM) 3,095 steps, 49.8 kernels, 1,307 us of kernels, about 45% of step time once a single 3.5 s gap between phases is set aside; MoE feed-forward 4,128 steps, 28.6 kernels, 1,327 us of kernels in a 1,437 us span (92% busy, 3-4 us between kernels), 26%; attention 1,032 steps, 90.9 kernels, 4,733 us in 5,586 us (85%), 25%. mul_mat_q a step: 531 / 830 / 629 us. Where the non-kernel time is: recurrent - __amd_rocclr_copyBuffer -> __amd_rocclr_copyBuffer 0.75 a step x 487 us (365 us a step: the device-host-device round trip of the small sums done on the host by cpu-root), mul_mat_q -> convert_unary_cont_vec4 0.88 x 223 us, copyBuffer -> k_bin_bcast 0.06 x 2.9 ms; attention - mul_mat_q -> convert_unary_cont_vec4 0.88 x 482 us; MoE - nothing above 11 us a step. So the recurrent step's host round trips are the loose part; the MoE step is tight and its cost is the matmul. mul_mat_q: 48,488 launches, 39.7% of kernel time on an XTX (51.2% on the R9700), 32 variants. MoE expert weights with mmq_x 16 (type 20 IQ4_NL grid 1280x3200 288 us a launch 16.0%; type 21 IQ3_S 64x3200 190 us 15.3%; 128x3200 351 us 14.5%) are about 46% of mul_mat_q time, about 18% of all kernel time; Q8_0 with mmq_x 128 (5120x16 88.6 us 11.3%; 1280x16 77.5 us 11.1%; 1536x8 stream-k 82.4 us 10.5%; smaller ones) about a third. Checked in the source: MMQ already uses the 8-bit WMMA instruction on RDNA3 and RDNA4 (__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32 and the gfx12 form in mma.cuh; AMD_WMMA_AVAILABLE is defined for both and selects the MMA data layout), so there is no dot-product path to replace for Q8_0. Needed before any matmul work: bytes of weights a launch per variant (the trace has no shapes; record mode has) to put each against the card's memory bandwidth, and a standalone bench of the expert variants at their real shapes, formats and tile width. Candidates then: fill the expert tiles (group tokens by expert, widen mmq_x when an expert has the tokens), and quantize + mul_mat_q in one launch (launch count, not kernel time). Qwen3.8-27B (65 blocks, 5,120 wide, dense, Q8_0, recurrent + attention): no expert family; not traced yet.



### Re-measured on pin b11552, and the premise corrected (2026-10-12)

Trace: Flash-Next 24K cold prefill on build b-pq11552 (run c48n-d24576, 38.7K tokens at 1379 t/s under rocprofv3), plus an 8K kernel + HIP API trace (run c48api-d8192). Tools: kernel-step-census.py (now with a median-step line, median-ranked pair gaps and an idle breakdown by gap size) and the new gap-api-calls.py (HIP calls the launching thread makes between two given kernels), on branch lab/qfp48-gap-api.

- The bump changed nothing in the step shape: 8351 steps per XTX, median 37 kernels a step (mean 44.1), mul_mat_q 39.4% of kernel time, flash_attn_ext_f16 17%.
- **"Only 62-64% of a step is kernel time" is a mean over the whole trace and is wrong as a steady-state figure.** The median step is 1330 us of kernels in 1480 us: the XTXs are 91-92% busy in a typical step (R9700 86%); 10th percentile 74%.
- Idle time inside steps, XTX 0, 7.98 s in total over the trace:
  - gaps under 10 us (356K of them, ~3.3 us each): 1.19 s, 15%. This is the launch overhead that fusing kernels can remove: at most ~4% of this prefill.
  - gaps of 1-100 ms (514 of them): 2.44 s, 30%. Almost all around buffer copies (copyBuffer -> copyBuffer 0.74 + 0.15 s, bc_dsv4_hc_post_gate -> copyBuffer 0.43 s, mul_mat_vec_q -> copyBuffer 0.30 s): the boundary between prompt batches, about 32 ms of idle per 512-token batch, ~8.7% of the prefill. In those gaps the launching thread is mostly in hipStreamSynchronize (about 27 calls, 2.2 ms each) and hipMemcpy2DAsync / hipMemcpyAsync (8K API trace).
  - gaps over 100 ms (6): 4.21 s, 53%. One 2.97 s wait before the first prompt batch (fillBuffer -> copyBuffer, start-up) and first-use code object loads (hipModuleLoad 120-320 ms each, e.g. the first batched half-precision gemm after mul_mat_q): about 1 s inside the first prefill of a server start, absent afterwards.
- The pair that looked like a 239 us per-occurrence host gap (mul_mat_q -> convert_unary_cont_vec4) has a median gap of 9 us; its mean was two code object loads. No per-step cost there.

Ranking for prefill time on this configuration: (1) the batch boundary, ~8.7%; (2) launch overhead, <= ~4%, which bounds what kernel fusion (megakernels) can return here; (3) first-use loads, ~1 s once per server start (a warm-up request would move it out of the first user request).

Next: attribute the 32 ms per batch boundary - which streams the synchronizes wait on (drafter / catch-up on the 6900 XT, the slower R9700, KV or input uploads), with 1359 (pipeline) and 1348 (deferred catch-up) on and off, on a warm server so start-up waits are out of the picture. Also seen: under rocprofv3 the server's shutdown hangs and is killed after the run (trace complete); not yet checked without the profiler.


## 2026-10-14 BCOP191: Q8_1 quantize -> MMQ producer-fusion admission (b11552)

**Disposition for this one candidate:** no generic producer/MMQ fusion patch or hardware queue. Keep QFP48 pending for its *other* trace-led candidates and batch-boundary attribution. This section supersedes the broad suggestion above to fuse every adjacent `quantize_mmq_q8_1 -> mul_mat_q` pair; adjacency does not prove a compatible producer, saved quantization work, or an end-to-end critical path. No BigCherry GPU experiment was performed in this audit.

### Actual dispatch and lifetime

- `ggml/src/ggml-cuda/ggml-cuda.cu::{ggml_cuda_mul_mat,ggml_cuda_mul_mat_id}` selects MMVQ, MMQ, MMF or cuBLAS based on batch, weight type and device. A graph-level `GGML_OP_MUL_MAT[_ID]` is not necessarily an MMQ launch.
- `ggml/src/ggml-cuda/mmq.cu::ggml_cuda_mul_mat_q` requires F32 `src1`, chooses precision via `ggml_cuda_mmq_get_prec_src1`, allocates per-call `src1_q8_1` from `ctx.pool()`, then launches `quantize_mmq_q8_1_cuda` (or `quantize_mmq_fp4_cuda`) **before** `ggml_cuda_mul_mat_q_switch_type` on the same stream. The dense path and routed `ids` path have different strides, allocation sizes and producer calls. The routed path runs `ggml_cuda_launch_mm_ids_helper`; when `ne11 == 1 && n_expert_used > 1` it uses `quantize_scatter_mmq_q8_1_cuda` and an inverse slot map. This is not a plain adjacent graph-node fusion.
- `ggml/src/ggml-cuda/quantize.cu::quantize_mmq_q8_1_cuda` selects **D4 / DS4 / D2S6** scale/sum layouts according to weight type. `block_q8_1_mmq` and its padding/stride are not interchangeable with the `block_q8_1` used by `quantize_row_q8_1_cuda` in MMVQ.
- Validated BigCherry `1307_q81_activation_cache_mmvq` already owns context-scoped stable-slab **MMVQ** Q8_1 caching and fused RMSNorm/activation/hyper-connection producers (former 1235/1309-1312). It is **not** a reusable MMQ producer for prefill; no second cache, graph dispatcher, allocation owner or quantization format should be introduced by QFP48. `1313_scale_act_fuse` and `1355_hc_post_gate_fuse` own their existing elementwise/gate fusions.

### Evidence and upper bounds, not a claimed gain

- Historical ub256 Flash-Next trace: `quantize_mmq_q8_1 -> mul_mat_q` appeared **5.12 times/step**; quantization averaged **5.6 us/launch**; ordinary short gaps were about **3.3 us**; the **mean** step span was **1,592 us**. A deliberately optimistic illustrative bound, assuming *all* those quantization launches and short gaps disappear with no overlap, is `5.12*(5.6+3.3)/1592 = 2.86%` of a step. It is **not** a measured end-to-end gain, and the 3.3 us is not a matched pair-specific gap.
- The later **b11552** trace is a different run: typical XTX step is 91-92% kernel-busy, and **all** sub-10-us gaps total only about 4% of prefill wall time. Its 24K quantization aggregate (1.37 s GPU kernel duration) is work still needed for MMQ, not 1.37 s of automatically removable wall time. **Do not multiply old per-step counts by new trace step counts.**
- The measured larger opportunity is batch-boundary copy/synchronization idle (~8.7% of the traced prefill), already under the QFP48/QFP50/QFP16 evidence path; no change to their current implementations or queued lanes is authorized by this audit.

### Upstream/fork comparison and decision

- [llama.cpp #29948](https://github.com/ggml-org/llama.cpp/pull/29948), inspected at `b6e0f03348b519d6118d5b66bff43d253385ada5` (open 2026-10-13), fuses **dense FFN gate/up matmuls plus GLU** in the MMQ writeback, reusing one Q8_1 activation quantization. Its code explicitly gates on `turing_mma_available`, excludes `ids`/MoE, checks quant precision, row/tile shapes and stream-K efficiency. It is **not** an AMD or Flash-Next routed-expert result; keep as a separate dense-only correctness/tiling reference, not a patch to stack here.
- [vLLM RMSNorm+quant fusion](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rms_quant_fusion.py) implements dtype/scale-matched graph patterns and ROCm FP8 producer kernels. The reusable idea is **typed producer/consumer matching**; FP8 scales and layouts do not transfer to GGUF Q8_1 MMQ. No SGLang/NVIDIA performance claim is admitted as gfx1100/gfx1201 evidence.

### Bounded reopening and implementation contract

**Stage 0 (no patch, reuse existing evidence):** On one warm, exact-revision Flash-Next production trace, join `kernel-step-census.py` and `mmq-variant-census.py` with existing `gap-api-calls.py` / backend dispatch receipts. Per physical GPU and same stream, record: graph op and consumer count; source weight type; `src1` type/shape/strides; `ids` and dedup/scatter route; `prec_src1` and D4/DS4/D2S6 layout; actual quant/MMQ kernel pair; bytes written/read; producer kernel; quant time; short gap; and **exposed** wall-time share. Separate prompt from MTP verification and dense from routed MoE. Fail closed if any required identity is missing. Do not reuse the older mixed-trace estimate as a promotion gate.

**Reject/terminate** this candidate if there is no single-consumer, F32, format-compatible dense MMQ producer, if quantization must still materialize for another consumer, if the measured **perfect-removal** end-to-end ceiling is <3%, or if the candidate relies on a routed `ids`/scatter map without a complete slot-preserving proof. Keep stock `ggml_cuda_mul_mat_q` and 1307.

**Only if Stage 0 clears >=3%:** one default-off, *existing-owner* 1307/MMQ extension may produce the exact `block_q8_1_mmq` layout from one proven producer, with no extra global cache/selector. Reserve scratch via the existing `ctx.pool()`; lifetime lasts through the consuming MMQ launch/graph replay, and any type, layout, view-offset, device, stream, consumer-count or pool-reservation mismatch falls back **before** writing output or skipping a kernel. Routed MoE is excluded from v1. The producer must still materialize its original F32 result when other consumers require it. Do not silently replace D4/DS4/D2S6 or FP4 precision.

**Validation:** first a host selector/shape/layout fixture, then patch composition + `test-backend-ops` dense `MUL_MAT` with negative `MUL_MAT_ID`/FP4/multi-consumer cases; repeated same-process requests, graph capture/replay and reallocation, ubatch 1/4/16/128/512, long context, greedy/full logits/KLD and MTP acceptance. Count actual quant launches and expected vs observed bytes; a missing MMQ/quant step without a valid fused replacement is failure. If safe, paired same-binary ABBA at 8K/24K/98K on dual gfx1100 and separately gfx1201; gfx1030 only if a supported route is positively demonstrated. Require >=3% E2E median gain with positive CI95 lower bound, no >1% decode/control regression, no extra VRAM/pinned-buffer growth beyond the measured reserve. Otherwise retire the candidate.

**Audit-only validation:** actual b11552 and BigCherry 1307 source assertions 14/14; disposable source-shaped host admission fixture 10/10 and `py_compile` passed; 2.86% is an illustrative arithmetic ceiling. No patch application, pytest, HIP build, GPU benchmark or performance gain was produced.

## Change Log

- 2026-10-10T13:53:50.737727+00:00 (created-by): Created by claude
- 2026-10-11T05:43:00.138967+00:00 (updated-by): Updated: section:notes
- 2026-10-11T07:31:20.295965+00:00 (updated-by): Updated: section:notes
- 2026-10-11T08:06:11.851254+00:00 (updated-by): Updated: section:notes
- 2026-10-11T20:31:34.192893+00:00 (updated-by): Updated: section:notes

## 2026-10-14 BCOP206: residual batch-boundary gap is here, not in QFP53

QFP53/1362 is terminal: coarser KV+k-pool padding produced 24/40 graph reuses and cut host graph build 7.1 -> 3.0 ms/batch, but Flash-Next prefill was neutral and shared-input wait rose 3.3 -> 17.0 ms. Dense 27B regressed and changed greedy output at 32K. Move the rejected graph-reuse proposal to `docs/planning/completed/patching-qwen-flash-next/QFP53.md`; do **not** implement QFP53 candidate B, because validated 1359 already queues the next graph while the previous batch computes.

QFP48 owns **only** a bounded warm per-batch HIP API/device trace join for the remaining ~6-7 gaps near copies. Record previous last kernel/event, upload/copy/sync, first next-batch kernel, graph UID, Meta rebuild, 1359 NextN fence and physical rank on correlated clocks. Separate overlapped host graph work from exposed wall time. At 1465 t/s a 512-token batch is ~349.5 ms, so +3% throughput requires >=10.18 ms *removable exposed* wall time. Reject without a new implementation if the joined trace falls below this gate; otherwise delegate one local default-off fix to the existing copy/wait owner (QFP50/1359 or QFP16/27), not a new scheduler/allocator. Preserve output/graph/event lifetime, multi-slot, MTP acceptance and numeric controls. Upstream llama.cpp #30453 (merged 2026-10-13) fixes Meta graph-slot reallocation after b11552 and is a prerequisite for any future persistent-graph candidate.

## 2026-10-16 BCOP239: retire the router split-K partial/reduce *fusion* candidate

**Disposition: ROUTER_REDUCTION_FUSION_NOT_ADMITTED.** This closes only the old "bc_moe_router_splitk_partial against its reduce" suggestion in the original candidate list. QFP48 remains pending solely for the separately owned warm batch-boundary trace join described in BCOP206; the Q8_1/MMQ producer candidate was gated in BCOP191. Do not create another router patch or hardware queue. QFP36/1357 owns the already validated split-K router, and its Flash-Next runtime profile currently enables it.

**Verified implementation path (BigCherry patch 1357 on pinned llama.cpp b11552):** src/llama-graph.cpp::build_moe_ffn marks a bare router MUL_MAT in op-param slot 6; ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat gates the marked, contiguous F32 AMD-HIP path on K>=512, experts>=64, tokens>=64 and BIGCHERRY_MOE_ROUTER_SPLITK=1. The owned ggml/src/ggml-cuda/bc-moe-router-splitk.cuh defines bc_moe_router_splitk_partial and bc_moe_router_splitk_reduce. A ggml_cuda_pool_alloc<float> scratch allocation belongs to the backend context pool and is used by both ordered launches on ctx.stream(); there is no new CPU/GPU transfer or cross-card collective in these two kernels. Meta and the existing graph/pool lifetime remain authoritative.

For the measured Flash-Next physical router K=2560, experts M=512, tokens N=512, PARTS=8, TILE=64: 512 partial workgroups write 8 MiB of F32 scratch, then 1024 reduction workgroups read that 8 MiB and write 1 MiB of logits. Scratch write+read traffic is 16 MiB per router invocation (logical minimum, excluding cache/replays); this is a byte count, **not** exposed wall time or a speedup ceiling. The partial workgroups each own one K partition; __syncthreads() cannot synchronize across their workgroups. Removing the second launch while retaining eight-way K parallelism needs a proven last-completer protocol with globally visible partial writes, per-tile counters/epochs, safe graph replay, pool reuse and cancellation. One workgroup computing all K parts instead changes occupancy and is a new GEMM algorithm, not an adjacent-kernel fusion. Fusing reduction with top-k crosses GGML operation/allocator ownership and may change expert selection.

**Existing measured evidence (not this audit):** 1357's 2026-10-10 Flash-Next same-binary four-request ABBA found +3.2% pooled 24K prefill, but changed greedy text; pooled decode was 69.8 -> 67.2 t/s with a single -20.4% request. The historical 1.93 s bc_moe_router_splitk_partial total is time spent doing the GEMM, **not** time removable by deleting its reduction. No matched current-pin reduce-kernel duration, launch gap or exposed critical-path share was found in the published QFP48 trace notes. A fused-kernel gain is therefore unquantified, not established.

**Upstream/fork code inspected, not portable evidence:** vLLM csrc/rocm/skinny_gemms.cu::wvSplitKrc_ uses a last-shard atomic counter and optional fixed-order scratch readback, plus per-stream workspace/graph-capture warmup in vllm/model_executor/layers/utils.py; it is gated to gfx950 and F16/BF16 skinny shapes (roughly 10-128 tokens), not BigCherry's F32 512-token gfx1100/gfx1201 router. vLLM's rocm_fp32_router_gemm.py is also gfx950-only, <=32 tokens and whitelisted K/expert shapes, not K2560/M512/N512. Merged llama.cpp #25952 fuses **post-expert weighted combination**, not the upstream router split-K partial/reduce; open #30553 caps MMQ Stream-K empty workgroups, a different kernel and ownership. Do not port or stack these as a router fix.
- https://github.com/vllm-project/vllm/blob/main/csrc/rocm/skinny_gemms.cu
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/utils.py
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/router/rocm_fp32_router_gemm.py
- https://github.com/ggml-org/llama.cpp/pull/25952
- https://github.com/ggml-org/llama.cpp/pull/30553

**Cheapest discriminator:** disposable host-only source-shaped fixture passed 9/9 cases: Flash-Next and other-MoE CTA/scratch accounting, MTP width fallback, shape/type/architecture/marker guards, order-sensitive F32 reduction, unfenced cross-CTA read and replay counter reset. These are logical counterexamples, not GPU/graph-capture tests. No repository pytest, HIP build, hardware execution or new benchmark.

**Reopen only if** a warm exact-pin, per-physical-rank rocprof/HIP API join shows the **reduce kernel plus its launch gap** (not the partial GEMM) accounts for >=3% *removable end-to-end* prefill wall time. Reuse QFP48's existing trace join and QFP36/1357 implementation ownership; require a default-off single-variant design with per-tile epoch/counter reset, global publication fence, graph replay and pool-lifetime proof, multi-request/multi-ubatch/cancellation tests, near-tie top-k and full-vocab/KLD/greedy controls, and expected-vs-observed GEMM/reduction work. On dual gfx1100 and gfx1201, promote only with same-binary ABBA CI95-low >=3% E2E and <=1% decode/control regression, no unexplained token or MTP acceptance changes; otherwise terminal REJECT_BELOW_EXPOSED_GATE / REJECT_NUMERIC / REJECT_LIFETIME. No P2P, RCCL or new scheduler/cache/dispatcher is authorized.
