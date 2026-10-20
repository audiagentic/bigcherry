---
id: BCOP164
order: 164
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-12T14:08:36+00:00'
created-by: agent
priority: P3
work: S
---

# PRBE54: reconcile historical Q5 KV hardware evidence before new campaigns

**Discovery:** Patch 1271 already has two b11126 Q5_0 GPU validation records in `evidence/validation.json`, omitted from the plan's later “no GPU measurement” statement. Both are INVALID (one session versus four required); gfx1201 pp512 +0.0143% CI95 [-0.2110,+0.2306] with full-vocab PASS, gfx1100 +0.7658% CI95 [-0.0309,+2.0003] with full-vocab FAIL (material diff 0.342 >0.02). Marker proves selector entry, not per-cell launch. No validated gain.

**Owner / consolidation:** PRBE54 and 1271 remain authoritative for contiguous Q5 FlashAttention F16 conversion. BCOP79 already owns Q5_1 arithmetic parity; no duplicate Q5_1 plan, patch, selector, scheduler, allocator or telemetry. Active QFP17/Radiance work remains untouched.

**Already acted on:** b11126 single-session gfx1100/gfx1201 validation and 2026-10-09 Q5_1 host-rounding audit. No current-pin hardware run or new patch this audit.

**Unresolved bounded action:** Add a read-only evidence classification regression and Q5_0/Q5_1 reference mechanics fixture; if a current production Q5 need is demonstrated, prove actual TILE/MMA converter launches, bytes and >=5% critical-path share on one current-pin isolated lane. Correct gfx1100 logits mismatch before any promotion. Do not queue the old four-session matrix now.

**Terminal gate:** If no actual Q5 production path, share <5%, Amdahl ceiling <3%, missing attribution or unresolved correctness: REJECT/retire 1271. Otherwise >=4 independent sessions/arch, >=10 ABBA pairs, full-vocab/KLD/MTP/graph parity, CI95-low >=3% E2E prefill and <=1% control regression. Default-off until fully qualified.

**Dependencies:** PRBE54 technical detail; BCOP79 Q5_1; upstream llama.cpp #27140 open, #29827/#29846 closed unmerged; QFP17 protected.
