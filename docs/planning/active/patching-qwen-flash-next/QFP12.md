---
id: QFP12
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T17:03:36.365732+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: S
---

# 1306 split the Qwen4Exp shared expert (mirrored upstream: full copy computed on every GPU)

## Description

GPT review req_07434b19f4534cb3 found that under -sm tensor upstream gives ffn_{up,gate,down}_shexp split axes only for DSV4; on Qwen4Exp they fall through to MIRRORED, so every GPU stores and computes the whole Q8_0 shared expert in every layer. Patch 1306 (BIGCHERRY_SHEXP_SPLIT=1) applies the DSV4 layout (up/gate axis 1, down axis 0, anchored to ffn_down_shexp.weight): each GPU computes ~1/3, its partial output joins the routed experts' partial sum before the same AllReduce (no new collective), and shared-expert VRAM per GPU drops to ~1/3. Same review: 1305's shexp rules were a no-op for this reason (documented), and the once-only log flags in 1303/1305 were a data race (fixed with std::atomic).

## Steps

1. Screen on profile v2 (240K f16) at ~24K and ~80K: SHEXP_SPLIT=1 vs 0, same binary. 2. Check acceptance/greedy (summation order changes). 3. Measure freed VRAM per GPU and re-run the max-context search. 4. Re-measure AR arrival skew (ar-boundary.py).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

ms/step ABBA; greedy/KLD vs mirrored; VRAM per GPU after load; activation via BIGCHERRY_PATCH_HIT patch=1306_shexp_split.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Related: RNX10 (shared-expert launch fusion), PRBE34/101 (shared expert on an aux stream - with the split, the per-GPU shared-expert work is smaller, changing that trade-off), QFP09 (per-class balance).

2026-10-04 screens (flashnext-v2-1306-d24k/-d80k, profile v2 240K f16, same binary): ~24K 47.2 vs 46.0/46.6 ms/step (+1.9%), acceptance 169/254 vs 175/240; ~80K 53.6 vs 53.3/53.4 (neutral), acceptance 166/266 vs 170/255. VRAM freed small: -185 MB per XTX, -135 MB R9700 (shared expert ~0.5 GB total). Activation proven (PATCH_HIT patch=1306_shexp_split). Verdict: neutral-to-negative; parked. The mirrored shared expert is small enough that the redundant compute is cheap, and splitting changes summation order (lower acceptance) and likely adds a per-layer combine around the sigmoid-gated shexp output. Not adopted.

## Change Log

- 2026-10-03T17:03:36.365732+00:00 (created-by): Created by agent
- 2026-10-03T17:21:20.454741+00:00 (updated-by): Updated: section:notes
- 2026-10-03T17:21:23.551814+00:00 (updated-by): Updated: priority='P3'
