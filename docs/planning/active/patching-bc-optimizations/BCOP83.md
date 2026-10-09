---
id: BCOP83
order: 83
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-09T04:13:01+00:00'
created-by: agent
priority: P2
work: S
---

# PRBE28: float-only GEMM row padding gate

## Discovery / disposition
Pinned b11474 loads packed GGUF with ggml_nbytes(cur) and linear upload. Arbitrary nb[1] padding is not a safe packed quantized GEMM optimisation. AMD-Ecosystem/llama.cpp PR #57 and upstream llama.cpp #26621 explicitly restrict original-weight row padding to F16/BF16/F32; Q8_0 and other quantized weights remain packed. No BigCherry performance improvement established.

## Owner
PRBE28 owns original float-weight row stride; PRBE29 owns quantized F16 shadows; PRBE30 owns shadow K-pad; PRBE31 owns BLAS crossover. No duplicate allocator or dispatch table. Last independent PRBE28/29/30/31 triage: 2026-10-08 06:47 UTC, outside 12 hours. Engine migration, MEN03/MEN05, Radiance, QFP, MTP and auxiliary-expert work protected.

## Gate
First measure eligible float-weight GEMM critical-path share and L2 aliasing on gfx1100/gfx1201. If none or theoretical E2E ceiling <=3%, close. Else adapt upstream #26621 default-off; validate packed-to-padded 2D upload, strides, mmap/async, graph, memory <=256 MiB/device, quant negative controls. Four sessions, ten ABBA pairs each; CI95-low >=3% E2E prefill, <=1% decode regression. Eight host layout fixtures passed; no GPU benchmark or build.

## References
https://github.com/ggml-org/llama.cpp/pull/26621
https://github.com/AMD-Ecosystem/llama.cpp/pull/57
