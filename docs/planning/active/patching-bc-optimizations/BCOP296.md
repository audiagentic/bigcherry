---
id: BCOP296
order: 296
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-18T06:10:30Z'
created-by: agent
priority: P2
work: S
---

# PRBE116: gate upstream Q3_K VDR2 on RDNA without inheriting Q2_K results

**Discovery:** Merged llama.cpp [#30517](https://github.com/ggml-org/llama.cpp/pull/30517) introduces Q3_K VDR=2 with two adjacent integer-dot chunks and shared packed-scale extraction. The pinned b11552 path is VDR=1. The upstream R9700 gfx1201 width1 Q3_K kernel 40.7 -> 28.2 us and Qwen3.5-4B Q3_K_S tg128 135.2 -> 142.1 t/s are **external measurements**, not BigCherry gains. The upstream generic two-row tile does **not** apply to RDNA. Host integer/scale mapping passed 16,384/16,384 comparisons; synthetic F32 grouping was non-bit-identical in 2,394/4,096 cases. No HIP/GPU build or BigCherry benchmark.

**Owner/consolidation:** [PRBE116](../patching-rdna-boost-experiments/PRBE116.md) owns Q3_K VDR2 only. Reuse 0600/0650 geometry/forced dispatch, stock Q8_1 producer and existing validation contracts; keep PKC05 Q2_K, rejected 1204 Q6_K, QFP10 IQ MMVQ, 1241/1274 F32 activation and RAD06 helpers separate. No new global selector, allocator, scheduler, profiling system, patch or hardware queue.

**12-hour gate:** selected `feat/rad06-profile2` from independent `a80aa61e` (2026-10-18 04:10 UTC). Visible commits/PRs since 2026-10-17 18:10 UTC advance RAD06 helper GEMV/WMMA/AHALF/VECIO/replay/profile, not Q3_K VDR2; those paths remain excluded. Off-repository queues not verified. New upstream #30517 is materially distinct from prior scheduled Q2_K and Q6_K audits.

**Unresolved action:** First prove a Q3_K_S route and >=3% optimistic E2E ceiling from **exposed** per-kernel wall time; if eligible, isolate Q3_K-only #30517 VDR2, verify 0600/0650 composition, integer/high-mask parity, F32/logits/KLD/MTP acceptance, exact work, graph replay and gfx1100/gfx1201 resource occupancy. Then >=4 paired sessions/architecture and CI95-low >=3% E2E with <=1% control regression.

**Terminal:** `NO_Q3_ROUTE`, `NO_EXPOSED_COST`, `BELOW_CEILING`, `NUMERIC_DRIFT`, `MISSING_WORK`, `GRAPH_LIFETIME_FAIL`, `NO_GAIN`, `REGRESSION` or `QUALIFIED_Q3_VDR2`. Until proven, retain pinned VDR1.
