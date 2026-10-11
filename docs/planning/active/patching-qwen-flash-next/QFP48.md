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

Fusion pays where two memory-bound kernels read or write the same data back to back. Candidates to check, each against the trace order of kernels (not assumed): quantize_mmq_q8_1 into its producer (rms_norm or the HC pre-grid), the two get_rows passes, k_bin_bcast into its neighbour, and the 1.93 s router partial against its reduce.

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

## Change Log

- 2026-10-10T13:53:50.737727+00:00 (created-by): Created by claude
- 2026-10-11T05:43:00.138967+00:00 (updated-by): Updated: section:notes
- 2026-10-11T07:31:20.295965+00:00 (updated-by): Updated: section:notes
- 2026-10-11T08:06:11.851254+00:00 (updated-by): Updated: section:notes
- 2026-10-11T20:31:34.192893+00:00 (updated-by): Updated: section:notes
