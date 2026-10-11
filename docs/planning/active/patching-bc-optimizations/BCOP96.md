---
id: BCOP96
order: 96
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-09T17:04:57+00:00'
created-by: agent
priority: P1
work: S
---

# PRBE111 / 1273: qualify IQ MMVQ VDR arithmetic before new hoist

## Discovery / disposition

Pinned b11474 already hoists IQ4_XS scale and IQ3_XXS aux32/ls outside the vec-dot inner loops; no blanket metadata-hoist implementation is justified. Evaluated 1273 splits the IQ3 truncating integer scale across VDR1 halves and the IQ4 F32 product across VDR2 halves, so unchanged quant math/bitwise identity is **not** established. Host-only fixtures: 25,344/67,600 IQ3 integer partitions differ (signed -32..32, ls 0..15); 51,459/131,072 synthetic IQ4 float32 reassociations differ. No GPU correctness failure, BigCherry IQ speedup or current-pin A/B was measured. Upstream HIP #27962 SWAR is already in b11474; external NVIDIA PRMT and Intel SYCL row-pairing are not RDNA qualification.

## Ownership / recent-work exclusion / novelty

PRBE111 owns any future IQ descriptor or hot-loop specialization; evaluated patch 1273 owns VDR/nwarps variants; 0600/0650 own MMVQ geometry, PRBE47/BCOP84 own Q6_K DPP. Last independent 1273 implementation work was 2026-10-03; the October 8 BPB01 reconciliation is >12 hours old. PRBE111's October 8 triage is outside the exclusion window. Recent BCOP84 concerns Q6_K wave32 DPP, not IQ VDR integer rounding. Protected active Flash-Next accuracy (1334/1347/1350), router 1357, Meta 1358, MTP, QFP35/41, Radiance and engine lanes are not edited or queued.

## Bounded action / terminal

First reuse 1273's existing package for a host fixture on real packed IQ3/IQ4+Q8_1 blocks, comparing stock and split integer intermediates and F32 output bits. If IQ3 VDR1 changes stock arithmetic, **reject exact-parity promotion** unless the algorithm is redesigned to aggregate integer sums before scaling; do not silently loosen greedy/MTP correctness. Next prove actual single-token IQ kernel activation and measured critical-path share on gfx1100/gfx1201 with rocprof; close PRBE111 if theoretical E2E ceiling <3% or compiler already hoists equivalent work. Only a surviving IQ4-only ISA-gated variant may run four-session, >=10 paired-round E2E tests with exact full-vocab/greedy/MTP controls, CI95-low >=3% and <=1% regressions. No new selector, allocator, dispatch table or experimental queue now.

## Evidence / sources

PRBE111; 1273 patch.py, README/SUMMARY; `releases/evidence/bpb01-four-evaluated.md`; pinned b11474 `mmvq.cu`, `vecdotq.cuh`, `vendors/hip.h`; upstream [HIP #27962](https://github.com/ggml-org/llama.cpp/pull/27962), [SYCL #30226](https://github.com/ggml-org/llama.cpp/pull/30226); [localweights vLLM GGUF fork](https://github.com/localweights/vllm-gguf-plugin) (README claims only, not code-verified/AMD measured). 14/14 source-static checks, 67,600 IQ3 host integer and 131,072 IQ4 host float32 cases; no repository pytest, build or GPU benchmark.
