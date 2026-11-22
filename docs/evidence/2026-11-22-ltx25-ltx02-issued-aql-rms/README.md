# LTX02 C05: native AQL graph execution and numerical proof

**Date:** 2026-11-22. **Result: PASS.** The actual patched Radiance 2.2/ABI44 LTX Q/K global RMS graph executes through RadCtx on two gfx1100 RX 7900 XTX cards and one gfx1201 R9700. This does not enable production TP3.

## Numerical results

| Visible physical device order | Native AQL exact BF16 | Exit | Whole process |
| --- | ---: | ---: | ---: |
| 0,1,2 (gfx1100/gfx1100/gfx1201) | 2,088,960 / 2,088,960 | 0 | 0.46 s |
| 2,0,1 (gfx1201/gfx1100/gfx1100) | 2,088,960 / 2,088,960 | 0 | 0.31 s |

Each run covers 16 cases: Q and K for video (4096 all-head width) and audio (2048 all-head width), rows 1/4/16/64, and both legal 8/8/16 and 12/12/8 head splits. Three distinct physical rank devices each issue two actual exact native collectives per case. The same native code and independent full-width FP32 RMS/BF16 oracle also passes through the host backend. BF16 gain values come from the real read-only IQ8R checkpoint. The input activations are deterministic synthetic BF16 values, not captured TP1 model intermediates. These end-to-end process times are **not** kernel timings or video-generation performance measurements.

## Reused execution and failure analysis

The sole test source is tools/lab/radiance/ltx-tp3/native-c01-gate/ltx_c05_issued_test.cpp; it compiles into the native host CTest and an **opt-in** HIP-enabled AQL executable. Both use the actual source-managed LTX2 patched translation unit, native RadBuilder, Registry, Program, RadCtx and the existing Tp3GlobalRms graph. GPU code uses native rank-devices, AQL streams, native allocations, actual gfx11/gfx12 operation rows, source BF16 gains, and Radiance's exact collective.

The initial host test failed (77/8192 exact) because externally pre-seeded Q/K inputs had transient memory-aliasing. The test fixture now keeps those inputs live across the step, without changing the production C05 graph. The first GPU attempt timed out before graph build while Registry scanned unrelated SDK plugins. The second isolated the existing libr3, libr4d and libref plugin libraries, reusing the C02 symlink staging pattern, but timed out in a sequential Builder::finish. Inspection of Radiance's native collective init confirmed an **init-time rank rendezvous**; finishing all three rank builders concurrently removed the deadlock. Both timeouts were negative diagnostics, never reported as successful numerical runs. No runtime collective change or second implementation was added.

The HIP-enabled core's CMake build receipt must match Radiance 2.2.0, the exact upstream source directory, and RAD_WITH_HIP=ON. The test uses the whole engine core with exported native AQL kernel registration and shared-slot symbols. Reverse device order resolves actual rank GPU architecture, not a fixed assignment.

**Raw run receipts:** stored outside Git in the task work root under ltx02-aql-c05-fullrows-proof/ and ltx02-aql-c05-reordered-proof/ (c05-native-aql-ctx.json and .log per run). Both ran under short exclusive acquired GPU leases with a 40-second subprocess timeout. The GPU lease is now free and ROCm reports no resident KFD processes.

## Acceptance boundary and next slice

This closes the seeded-input, real-checkpoint-gain C05 numerical issue gate over 1/4/16/64 rows and reversed device ordinal. It does **not** close LTX02 as a whole: a real TP1 checkpoint activation comparison, the complete IQ8R C06 self-attention block (fused QKV, RoPE, attention, gate, output reduction and one post-reduce bias), missing-rank/timeout failure controls, and model video-quality and serving admission remain open. RUM06 owns generic rank lifecycle; RUM13 owns model-independent AQL qualification. Keep TP3 production refusal active.
