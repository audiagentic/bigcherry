---
id: BCOP66
order: 66
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-08T11:06:24+00:00'
created-by: agent
priority: P2
work: S
---

# PHA08: b11474 HIP D=72 CLIP FA byte-range gate

## Discovery and change

Pinned b11474 b9acf138 and upstream master have identical fattn-tile.cuh. HIP copy width is 16 B; 18 source-derived host vector/row fixtures found no partial D=72 Q/K/V/output vector in contiguous paths. This removes BCOP25's speculative tail-guard-first experiment, not the unresolved aperture fault. Upstream #28608 is open, #28664 unmerged. No first-party new hardware or performance measurement.

## Authority, prior work and overlap

PHA08 owns instrumentation, safety and any narrow CLIP fallback. BCOP25 is completed as superseded. Existing CUDA/HIP FA, mtmd CLIP and HIP-autotune own kernel/dispatch/build. No duplicate scheduler, allocator, kernel clone or global FA disable. Active QFP/PA/MTP/MMQ/PRBE work remains untouched.

## Bounded acceptance / terminal disposition

Reproduce b11474 on 1x/2x gfx1100 with D=64/72/80 and 1024-2560px; record exact tensor ne/nb, allocation and last-byte access. No reproduction after full stress matrix -> close without patch. Reproduced fault -> localize first bad access/lifetime, then only a minimal fix or compile-verified HIP+CLIP+D72 matmul fallback. Safety requires 20 high-resolution repetitions, parity, graph/multi-request tests; kernel performance promotion requires >=5% measured VLM encode gain over safe fallback and <=2% healthy-control regression. Otherwise document blocked or safe fallback. No tail-guard experiment absent contradictory runtime evidence.

## References

PHA08; upstream https://github.com/ggml-org/llama.cpp/issues/28608 and https://github.com/ggml-org/llama.cpp/pull/28664; vLLM ROCm attention backend is a policy comparison only. All technical details live in PHA08.
