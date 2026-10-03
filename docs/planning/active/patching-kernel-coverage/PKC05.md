---
id: PKC05
order: 0
plan: patching-kernel-coverage
state: pending
created-at: '2026-10-01T05:39:29.192524+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
---

# Q8_0 decode and prefill headroom on dual gfx1100 (27B): profile, then MMVQ aligned loader / geometry / MMQ config

## Description

27B Q8_0 plain decode is ~36 t/s (38.4 with adaptive AllReduce) against a ~66 t/s DRAM roof: 27.8 ms/token vs 15.2 ms ideal. dev-gpt-agent estimate (2026-10-01, unverified) of the 12.6 ms excess: Q8_0 MMVQ memory/geometry 38-48%, launch/serialization gaps 18-24%, AllReduce 14-20% before adaptive (3-8% residual after), GDN 10-16%, attention 4-8%. Prefill pp4096 estimate: MMQ 60-70%, AllReduce 8-15%, GDN 10-15%, FA 5-8%.

## Steps

1. rocprofv3 attribution (tools/lab/profiling/profile-27b-q8.sh, queued as prof-27b-q8-1): per-device critical-path time per category, per token.
2. P1: Q8_0/F32 MMVQ cooperative aligned loader (8-block supertile, 272 B = 17x16 B, wave-coalesced loads; keep 1241 semantics). Expected +5-10% decode.
3. P2: sweep gfx1100 Q8_0 n=1 geometry 4x1/8x1/4x2/8x2 via the existing explicit MMVQ geometry machinery; bake only the winner. Expected +1-4%; drop if P1 dominates.
4. P3: profile ub2048 J distribution, sweep RDNA3 Q8_0 MMQ tile configs, add measured winners to mmq-config-rdna3.cuh. Expected +3-6% pp4096. If GDN >= 10-12% of prefill, prioritise chunked GDN first.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

vendor/llama.cpp/ggml/src/ggml-cuda/mmvq.cu, mmq-config-rdna3.cuh, gated_delta_net.cu (via new patches); tools/lab/profiling/

## Validation

Profile attribution recorded per category. Each patch: experiment contract on tierL-qwen27b-q8 -sm tensor, 4 sessions x 10 rounds, improvement CI > 0, control within 1%.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-01T05:39:29.192524+00:00 (created-by): Created by agent
