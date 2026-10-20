<!-- History (2026-10-20): the SUMMARY of a seeded but never-built engine patch package, rad0004_two_half_prefill, from the unmerged branch feat/rad04-two-half. The ID rad0004 was later reused for the remote-MoE diagnostics. Kept for the idea and its reasoning. -->
# rad0004_two_half_prefill

Correctness-first two-half prefill scheduling for qwen35 FP8.

With `RAD_TWO_HALF=1`, pure prefill steps with at least 1024 tokens issue each trunk layer as
A.mix, A.ffn, B.mix, B.ffn over A=[0,T/2) and B=[T/2,T). Decode, mixed steps, and other
architectures keep the existing path.

Slice 1 keeps one queue. Slice 2 will move A's fused sum to lane 1 with event record/wait.

Bit-exactness rules:
- Per-row GEMM, norm, quant, activation, residual and all-reduce arithmetic keeps the same
  reduction order inside a row.
- Norm/quant and collective fusion selection is made from the original full-step T, not a half's
  row count, so half boundaries cannot select a different kernel or fused collective.
- Dense paged attention keeps the original full cu_seqlens, block table and seqused metadata;
  a kernel row-half selector suppresses writes outside the requested half without rebasing causal
  positions.
- GDN keeps original 64-token chunks. A chunk crossing T/2 belongs to A so KKT/scan arithmetic is
  unchanged; B resumes from the next original chunk. The scan hand-off uses the ordinary FP32
  recurrent state cache. Half mode explicitly rejects FP16 recurrent state because writing and
  reloading it would add an extra rounding point.
- M-RoPE row slicing is not enabled by this package; half mode fails the step rather than silently
  changing component-major position strides.

The package is pinned to Radiance 89cee7ce414486c717aa2085c99e1f522e4c0ea6.
