# rad0061 — IQ8R FP32 scale-product diagnostic

## Summary

**Diagnostic only; not selected by production recipes.** Preserve the exactly
representable FP32 product of the IQ8R per-256-K U8 scale and per-row BF16 scale
through native Radiance GEMM. This updates the existing group-256 scale relayout,
narrow/K-parallel decode and prefill readers. The existing gfx1100 `libr3`
native rewrite uses the same modified `libr4d` source as gfx1201; no new
GEMM, weight loader or collective is introduced. `libref` remains FP32.

## Outcome

Fully rebuilt native XTX and R9700 plugins issued the existing LTX-2.5 C06
checkpoint-backed AQL attention graphs across three GPUs in both rank orders:
four video/audio × head-split cases passed in each order, 108 native instances
retired per order. Against the unchanged CPU reference, the IQ8R QKV
projections matched all 36,864 BF16 elements exactly and 12,287 of 12,288
final outputs matched. The two GPU orders agreed bit for bit.

The corrected native R9700 IQ8R selftest passed 35/35 existing cases against
the *FP32* scale fold without relaxing the numerical threshold. All 14
source-anchored edits reproduce the candidate exactly and are idempotent;
23/23 focused Python checks pass.

## Detail

The scale layout changes from BF16 to F32 only for IQ8R group-256 weights.
The original INT8 weight codes, activation quantization, BF16 output and
other scale formats remain unchanged. The new layout tag must invalidate
old cached plans and relayout buffers.

This is a runnable **diagnostic candidate**, not a production TP3 admission:
M=16/64 and real TP1 activation comparisons, VRAM and loader/refill admission,
cache invalidation, cancellation and performance remain required. The LTX TP3
production refusal is not changed. See
`docs/evidence/2026-11-22-rum15-f32-native-candidate/README.md` for the
actual checkpoint test, source and binary provenance and measured differences.
