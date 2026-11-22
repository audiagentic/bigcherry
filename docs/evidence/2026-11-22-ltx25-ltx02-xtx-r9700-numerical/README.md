# LTX02 C06 — isolate 7900 XTX / R9700 numerical differences

**2026-11-22. Result: no XTX-versus-R9700 numerical difference observed on the tested model-backed C06 block. CPU-reference versus AQL divergence first occurs at the IQ8R QKV projection, not at the rotated-INT8 input quantizer. TP1 parity remains open.**

## Measurement and controlled variable

The existing source-backed `ltx_c06_instancing_aql_test` was extended with an **opt-in diagnostic capture** (`BC_LTX_C06_CAPTURE_STAGES=1` with the existing `BC_LTX_C06_CAPTURE_DIR` and `BC_LTX_C06_CAPTURE_TAG`). It uses the already-issued Radiance C06 `RadCtx` graph, real LTX-2.5 IQ8R RAD2 checkpoint weights and **the same seeded BF16 activation** for video and audio block 0; nothing in the production kernel, patch or collective changed.

Two explicitly masked placements were executed under exclusive GPU leases, with the three native rank threads and the normal exact collectives:

| Mask (`ROCR_VISIBLE_DEVICES`) | Rank 0 | Rank 1 | Rank 2 | Result |
| --- | --- | --- | --- | --- |
| `0,1,2` | gfx1100 (XTX) | gfx1100 (XTX) | gfx1201 (R9700) | PASS |
| `2,0,1` | gfx1201 (R9700) | gfx1100 (XTX) | gfx1100 (XTX) | PASS |

Thus **ranks 0 and 2** process the **same IQ8R checkpoint shard** on each GPU architecture, while rank 1 is an XTX-to-XTX control. Both supported head partitions (8/8/16, 12/12/8) and both modalities were exercised. Every run retired all 108 native instances and exited 0. The lease was released.

The test-only capture copies each stage **at the actual issue point, before the normal memory planner may reuse the buffer**. Snapshots use native `RadCtx::buf_ptr`, existing stream and `rad_memcpy_async` on device, with a diagnostic synchronization. The same capture works on the CPU reference backend. No second math implementation or standalone benchmark runner is added. **Synchronization perturbs timings; this is a numerical experiment, not a speed benchmark.**

## Bitwise AQL XTX-versus-R9700 comparison

| Captured tensor | Exact values across GPU placements | Maximum absolute error |
| --- | ---: | ---: |
| Rotated INT8 activation codes | 36,864 / 36,864 | 0 |
| FP32 activation scales | 144 / 144 | 0 |
| BF16 QKV after fused projection | 36,864 / 36,864 | 0 |
| BF16 gate projection | 384 / 384 | 0 |
| BF16 QKV after global norm + RoPE | 36,864 / 36,864 | 0 |
| BF16 attention after head gate | 12,288 / 12,288 | 0 |
| **All captured BF16 intermediate values** | **86,400 / 86,400** | **0** |

Full post-reduction output was also **exact across GPU orders for all 12,288 BF16 elements** (4,096 video plus 2,048 audio per head split), consistent with the earlier LTX02 issue receipt.

The gfx1100/gfx1201 code-object distinction did **not** change measured C06 numerics for these shaders, head partitions and one-row input. This cannot establish performance parity, all-row parity or TP1 model quality.

## CPU reference versus native AQL

The same captured quantizer outputs also matched **CPU host to GPU exactly**: all **36,864 INT8 codes** and **144 FP32 scales** are bitwise identical. Gate projection was likewise exactly identical.

**The first difference is at the output of the IQ8R fused QKV GEMM.** For the same model and shard, native AQL QKV differs from the host `libref` output by relative L2 **0.2545%–0.3096%** over the 12 rank/modality/head-partition cases; the post-normalization/RoPE stage differs by **0.2541%–0.3201%**, and local attention after gating by **0.2307%–0.3038%**. These are measured errors, not approved tolerances.

Splitting the IQ8R QKV output by Q/K/V identifies nonzero differences in all three banks. On the 8/8/16 partition, the maximum BF16 absolute difference is **0.03125 video** and **0.0234375 audio** for Q; K/V maxima are **0.015625**. The other partition reconstructs the same full-head Q/K/V projection, as expected. This rules out activation quantization and the gfx1100-versus-gfx1201 choice **within this test**. The remaining causes to distinguish are CPU reference versus GPU GEMM accumulation/rounding and kernel-selected weight relayout; it is not sound to pick one without a TP1 or higher-precision reference.

The final checkpoint-backed output had AQL-versus-host relative L2 **0.330%–0.422%** and cosine **0.999991–0.999995** in the previous LTX02 receipt. Separately, host outputs from the two legal head partitions were **not bit-exact** (video 1,605/4,096 and audio 736/2,048 BF16 elements changed). No tolerance has been invented or waived based on these observations.

## Verification and retained data

- Rebuilt frozen Radiance 2.2 / ABI44 host and HIP-enabled native C06 fixture with original `libref/libr3/libr4d` kernel libraries.
- All **10/10 native host CTests passed** after compacting BF16, INT8 and FP32 capture into **one** reusable test-only function.
- **13/13 focused Python real-model/source tests passed**, no skips.
- Both AQL mask permutations pass with the final compact capture implementation (process exits 0), including a replay after rebasing on the merged C06 coordinate-allocation cleanup (PR #1316); source/checkpoint and production TP3 refusal unchanged.
- Binary tensor snapshots and bounded-process JSON receipt are in the task's external work root, **not Git**; they are evidence for reproducibility but not a distributable TP1 golden oracle.
- The capture is off unless explicitly enabled and adds **no production-code path**. It should never be used to claim performance timing; normal operation has no diagnostic synchronization.

## Correct next numerical gate

Capture a **single-rank TP1**, or another approved higher-precision, reference for the **same block-0 IQ8R checkpoint and exact input**. Compare Q/K/V GEMM outputs and, if necessary, inspect the native layout callback versus the CPU reference at one element/accumulator at a time. Then repeat with **real TP1 activation captures** and 16/64 rows before adjudicating numerical tolerance and any kernel fix. Generic loading/admission belongs to RUM14/RUM owners; LTX02 must not introduce another loader, quantizer, or collective.

**Do not change the XTX or R9700 kernels solely to force equality with the CPU reference: the two GPU architectures are already bit-exact for the measured case.**
