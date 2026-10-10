---
id: BCOP111
order: 111
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-10T08:05:23+00:00'
created-by: agent
priority: P2
work: S
---

# PRBE111: IQ3 split-VDR rounding gate; retire duplicate metadata hoist

## Discovery / disposition

Pinned b11474 already hoists IQ3_XXS/IQ4_XS invariant scales outside vec-dot loops. 1273 owns the existing IQ VDR/nwarps experiment; no second descriptor, dispatcher or metadata-hoist implementation. Its IQ3_XXS VDR1 rounds each integer half separately, while stock VDR2 rounds the combined integer sum once. A disposable compiled C++ test of 100,000 synthetic half-sum/scale tuples found 37,433 differing outputs (max |integer delta|=1). This is **not** a packed-block, GPU or logits result. Keep 1273 evaluated/unqualified and default-off.

## Owner / 12-hour exclusion

PRBE111 owns hoist/descriptor disposition, 1273 VDR/nwarps, 0600/0650 geometry, 1241 F32 activation, PRBE47 Q6_K DPP, QFP38 other MMVQ dispatch. Last independent 1273 disposition 2026-10-08 15:31 UTC; PRBE111 plan 2026-10-05. Both outside 12 hours. Current Radiance MXFP4, Flash-Next/QSA/router/MTP, QFP35/36/41/43 and collectives protected and unchanged.

## Next gate / terminal outcome

First valid packed-IQ3_XXS/Q8_1 host parity oracle (sign/scale/tails). If VDR1 differs from pristine, **reject VDR1** before GPU speed tests; only an integer-pre-round merge could restore equivalence. IQ4 VDR2 and nwarps-only are separate. Surviving arms require actual launch accounting, full-vocab/greedy/MTP/graph/multi-request parity, current-pin gfx1100/gfx1201 ISA/profile, and E2E contribution >=2.913%. Otherwise close. Promote only with >=4 sessions/arch, >=10 paired ABBA rounds/session, CI95-low >=3% E2E decode, <=1% controls. No duplicate queue, registry, cache or allocator.

## Evidence

Pinned b11474 `mmvq.cu`/`vecdotq.cuh`; 1273 patch/README/SUMMARY, BPB01; upstream llama.cpp #27828 (sm_60 only), vLLM GGUF and SGLang #35019 (not transferable AMD IQ decode). 11/11 source checks and 100,000 compiled host arithmetic cases ran; no repository pytest, packed-quant test, HIP build, model execution, profiler or hardware benchmark.
