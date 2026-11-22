# RUM15: native IQ8R FP32-scale candidate (diagnostic only)

**2026-11-22. Source-backed native F32 IQ8R scale candidate RUNNABLE and numerically verified on the XTX/XTX/R9700 three-card fleet, both rank orders. Production unchanged; model TP1 capture and memory/performance/admission remain open.**

## Change boundary

The original IQ8R RAD2 container stores **INT8 codes**, a **U8 per-output-row/per-256-K scale**, and a second **BF16 per-output-row scale**. Both original factors multiply to an exactly representable **FP32** value. The stock `libr4d` decoder rounded the product to BF16 before GEMM, unlike `libref`'s FP32 reference. The original numerical finding and independent 198-checkpoint-element oracle are recorded in `docs/evidence/2026-11-22-ltx25-ltx02-iq8r-scale-root-cause/README.md`.

New unselected diagnostic patch `rad0061_iq8r_f32_scale` changes the **existing** Radiance `libr4d` scale layout and consumers:

- IQ8R-only group-256 layout `r4d.iq8r.s.f32.t16` stores **FP32** group scale, tile-major, not a prematurely rounded BF16. The native source layout hook multiplies the original two source levels in FP32.
- Existing narrow GEMM, split-K/K-parallel GEMM and prefill fold use those scales as FP32; other I8 (group-128), FP8 and IQR layouts keep their original scale representation.
- The original `r4d_selftest` IQ8R oracle is updated to expect the **higher precision**, with the same existing error bounds. **Nothing changes the CPU libref reference, IQ8R codes, output BF16, collective, loader, runtime or production activation path.**
- gfx1100 uses the existing BigCherry `libr3` native rewrite **from the same updated `libr4d` source**; no second XTX implementation and no LTX-specific kernel.

The patch is `state = "diagnostic"` and **not selected into any serving recipe**. The native layout tag/element width changes, so this candidate cannot reuse serialized old layout plans. Production's world-size-3 LTX TP3 refusal remains enforced.

## Real-checkpoint mixed-GPU execution: **PASS**

With the full native gfx1100 `libr3.so` built by the standard BigCherry native rewrite from the same candidate source, and the fully rebuilt gfx1201 `libr4d.so`, ran the existing **C06 issued attention** test under an exclusive GPU lease. The actual LTX2.5 IQ8R checkpoint supplied all QKV/gate/output weights, the two rank-sharded real tensor layouts were materialized through each backend's own native relayout callback, and the original Radiance `RadCtx` graph issued Q/K normalization, RoPE, dense attention, output reduction and bias.

| GPU mask | Physical rank GPUs | Real checkpoint C06 test | Native instances |
| --- | --- | --- | --- |
| `0,1,2` | XTX / XTX / R9700 | video/audio × 8/8/16 and 12/12/8 all **PASS**, exit 0 | 108 retired |
| `2,0,1` | R9700 / XTX / XTX | same four cases all **PASS**, exit 0 | 108 retired |

Both runs exited zero and released the lease. The complete two-order outcomes are **bit-for-bit equal**, including every captured quantizer code/scale, projection, post-norm/rotary QKV, attention and BF16 final output. One-time wall times 0.739 and 0.619 seconds include startup and diagnostic transfer/synchronization and are **not speed measurements**.

Compared against the **unchanged pinned CPU `libref`** issuing the same real checkpoint and deterministic BF16 inputs:

| Stage | Exact CPU/GPU values | Maximum absolute difference |
| --- | ---: | ---: |
| Input INT8 activation codes | 36,864/36,864 | 0 |
| Input FP32 activation scales | 144/144 | 0 |
| Fused IQ8R QKV projection | **36,864/36,864** | **0** |
| Gate projection | 384/384 | 0 |
| QKV after global norm + RoPE | 36,846/36,864 | 0.00390625 |
| Attention after gating | 12,288/12,288 | 0 |
| Final BF16 video 8/8/16 | 4,096/4,096 | 0 |
| Final BF16 video 12/12/8 | **4,095/4,096** | **0.015380859375** |
| Final BF16 audio 8/8/16 | 2,048/2,048 | 0 |
| Final BF16 audio 12/12/8 | 2,048/2,048 | 0 |

All 12,288 final BF16 positions per rank placement are identical between the two GPU orders, and **12,287/12,288** final positions match the CPU reference. The single mismatched video element has overall case relative L2 **3.92e-5** (0.00392%), down from previous 0.330–0.422% GPU-versus-host C06 final output discrepancy. The tiny post-norm and one final output differences remain numerical findings, **not acceptance thresholds** and not yet TP1 whole-model activation parity.

Actual candidate plugin SHA256: gfx1100 `e6cd43e1363327c10643953ad5a9fe8a9c5e6be151e9151f8ad9fb5b60953e18`; gfx1201 `9c38079be3b0fb9f316ee5ba364ea402c6cd160c9bc67e44cfd59b26f6f1f906`. Existing C06 test binary SHA256 `33d8215f2330adbbc8f950d2120bafa25f2ece6f2971056664fdf14c2837a77c`. Captures and original external receipt are outside Git.

## Completed verification

- Native registered patch `FilePatch` source-equivalence: all **14 anchored edits** apply to the frozen pinned source, produce byte-identical candidate source and become `already-applied` on exact replay (**14/14**; layout 2, GEMM 5, prefill 3, selftest 4).
- Python precision, original LTX IQ8R numeric/plane/shard and patched-builder/reference tests: **23/23 pass**, using real LTX 2.5 IQ8R container for the model-backed subset. No error threshold was changed or bypassed.
- Native C++ layout object and gfx1201 HIP objects `r4d_gemm_fp8a8.hip` and `r4d_gemm_iqr.hip` compile. A **full from-source gfx1201 `libr4d.so`** and modified native `r4d_selftest` compiled and linked against pinned Radiance 2.2/ABI44 and ROCm 7.2.4.
- Exclusive-leased Radeon Pro R9700 native IQ8R selftest (`--case gemm_iq8r_nt_m16`, device mask restricted to the R9700): **35/35 PASS**, process exit 0. It includes K-parallel decoding, M=1/5/7/16/28/58/64, tiled M=200/300, K as large as 17,408, group-256 Hadamard quantizer and fused group-256 producers.
- The older BF16-folded selftest oracle produced one failed 2,048-by-10,240 decode cell (relative L2 against its **wrong BF16 fold** 0.003963); with the corrected **FP32 fold**, the same cell passes at 0.00165. The other reported cells similarly have FP32 exact-fold error typically **0.0015–0.0017**. The existing unchanged cutoff is 0.003. This is **not** FP32 full-model video parity.

- Complete standard native gfx1100 `libr3.so` build passed after restoring the already-managed `bc_ar_lifecycle_impl.h` source composition. This is the same RUM06 all-reduce seam as the pinned production plugins, **not** a new RUM15 transport or shim.

All diagnostic library and raw-output files are isolated in external work roots. No production binaries or installed prefixes were replaced, and no GPU process runs without `agent-lease`.

## Reproduce the native runnable case

After composing the pinned source, diagnostic patch and normal native gfx1201 plugin build, use the existing upstream executable:

```sh
ROCR_VISIBLE_DEVICES=2 \
  r4d_selftest "$RUM15_LIBR4D_SO" --case gemm_iq8r_nt_m16
```

Run through the host's exclusive GPU lease facility. The visible-device mask above is the **physical R9700** for the recorded fleet, not a general device ordinal guarantee; check the actual roster before running. This is a kernel selftest, not LTX TP3 serving admission.

The model-backed C06 three-rank validation (once both native plugins compile) uses the **existing** program and no new runner:

```sh
ROCR_VISIBLE_DEVICES=0,1,2 "$LTX_C06_NATIVE_AQL_TEST" "$LTX_IQ8R_RAD" "$RUM15_KERNEL_DIR" --c06-issued-aql
ROCR_VISIBLE_DEVICES=2,0,1 "$LTX_C06_NATIVE_AQL_TEST" "$LTX_IQ8R_RAD" "$RUM15_KERNEL_DIR" --c06-issued-aql
```

Both commands were executed under one exclusive GPU lease, returned 0, and their BF16 tensors were compared against the same pinned host reference as detailed above. The pinned test binary and candidate plugin hashes are recorded in this receipt section.

## Remaining gates (do not claim production readiness)

1. **Three-card C06 issued at M=1 is now green**; extend the actual production-oriented model input to **16/64 rows and real TP1 activation captures** and investigate 18 post-norm BF16 elements plus the one final BF16 element that still differ. No invented tolerance.
2. Confirm native format tag and layout fingerprint, reject an old BF16 stored-scale buffer with the new plugin, and run production loader/refill/SSD/VRAM tier tests (RUM14 owns generic loader).
3. Measure exact extra scale memory against three-card GPU admission. The entire 1,306-matrix IQ8R source has 127,309,824 group scales; F32 increases resident scale bytes by approximately **242.82 MiB if all scale groups were simultaneously resident**, not necessarily each device's usage. Run uncaptured K-parallel and prefill timings on XTX and R9700.
4. LTX02 owns real TP1 activation capture, complete quality parity, multirow 16/64 attention and fault/cancel/restart. No claim of full DiT, video quality or production release.
