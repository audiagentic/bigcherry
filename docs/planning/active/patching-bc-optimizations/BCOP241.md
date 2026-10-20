---
id: BCOP241
order: 241
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-16T08:09:29+11:00'
created-by: agent
priority: P1
work: S
---

# MET08: fail-closed offline expert placement before weighted-traffic experiment

**Discovery:** `expert-place.py` mutates copied GGUF shards before completeness validation. The `gate_up` shortcut admits incomplete fused layouts despite 1283 only supporting separate gate/up/down; duplicate names also pass count. Source-level integrity risk, no corrupted GGUF observed. The proposed 2:2:1 `--traffic` controls ranked expert assignment, not measured GPU service or throughput.

**Ownership:** MET08 offline writer and placement performance; MET11 B2 load-time mapping; MET09 router first divergence; MET04/1283 EP execution. MET07's upstream cache is separate. Corrected stale 1283 SUMMARY. No new writer/allocator/cache/solver/patch or queued hardware work.

**Eligibility:** `main` at independent `08441aa4` (2026-10-15 05:21 UTC). MET08 last independent placement 2026-10-07, MET11 last substantive clarification 2026-10-12; no selected-path commits or PRs in checked 12h window ending 2026-10-15 21:09 UTC. Recent BCOP239 router fusion is a different mechanism. External hardware queues not independently accessible.

**Action:** Fail-closed profile/tensor/shard checks, staging plus atomic publish, inverse-map/byte/hash manifest and negative host fixtures first. Then held-out 1:1:1 vs 2:2:1 work/service bound using existing 1338/routing-balance receipts. Only if >=3% exposed E2E opportunity, perform fixed-work MTP-off/on multi-request gfx1100/gfx1201 qualification after MET09 first divergence.

**Terminal:** `UNSUPPORTED_LAYOUT`, `INCOMPLETE_ARTIFACT`, `INVALID_PROFILE`, `NO_ADMISSIBLE_GAIN`, `NUMERIC_OR_WORK_MISMATCH`, `NO_NET_GAIN` or `QUALIFIED_OPT_IN`. Promote only with exact executed work, full logits/KLD/greedy, physical VRAM and CI95-low >=3% E2E, <=1% other-regime regression.

**Validation:** 12/12 source assertions, 7/7 host-only tests, Python compile PASS. No real GGUF copy, repository pytest, HIP build, GPU run or new benchmark. Full design and sources remain in MET08.
