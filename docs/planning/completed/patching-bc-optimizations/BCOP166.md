---
id: BCOP166
order: 166
plan: patching-bc-optimizations
state: deprecated
created-at: '2026-10-12T16:12:00+00:00'
created-by: agent
priority: P2
work: S
---

# PRBE67: retire unproven HIP FA graph-UID recapture

## Description

**Discovery/disposition:** The b11474/b11552 CUDA graph cache uses first-node identity only for lookup; its full node/source comparison is bypassed only when a nonzero graph UID matches. Ordinary scheduler graph splitting and Meta subgraph rebuilds generate new UIDs. The earlier proposed FA-only fingerprint is weaker than the existing comparison. Same-UID property mutation is a conditional host-model failure mode, **not** an observed gfx1201 fault. PRBE67 closed without a patch, hardware queue or new cache/selector.

**Ownership/consolidation:** PRBE67 is the sole historical HIP FA graph-UID hypothesis; PRBE66 is completed/merged. RD73/1233's alternative shape-key was rejected on controlled dual-XTX performance evidence; QFN03/1302 owns validated graph-instantiation OOM recovery; QFP06/1304's LRU cap is rejected. Do not reopen these mechanisms through a neighbouring plan or touch active Radiance/QFP15/release work.

**Evidence:** 13/13 upstream source-static checks and 8/8 disposable host control-flow cases passed; no BigCherry pytest, HIP build or GPU measurement. HI162's rejected RD73 tg128 delta -1.96% is historical and **not** a PRBE67 result. Current upstream draft #29287 is RPC graph caching, not a local HIP fix.

**Terminal / conditional gate:** CLOSED now. Reopen only on current-pin graphs-on failure absent with graphs-off **plus** same nonzero UID and changed FA source properties at the failed transition. Use existing graph/UID receipts; then prove a deterministic regression and fix the smallest violated UID lifetime or fall through to the existing full comparison. Require logits/MTP parity, multi-request/ubatch, long context, no stale replay, bounded graph memory/recapture and stable-control performance. Otherwise retain upstream. Full technical conditions live only in PRBE67.

**References:** PRBE67, PRBE66, RD73/1233, HI162, QFN03/1302, QFP06/1304; https://github.com/ggml-org/llama.cpp/pull/29287 .

## Steps


## Detailed Solution & Technical Design


## Code Samples & Guidance


## Files


## Validation


## Effort & Risk


## Standards


## Acceptance Criteria


## Notes


## Change Log

- 2026-10-23: state 'done' (invalid) corrected to deprecated
