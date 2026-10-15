---
id: BCOP205
order: 205
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-14T08:09:33Z'
created-by: agent
priority: P3
work: S
---

# QFP52: gate duplicate NextN D2H on actual exposed copy and output-lifetime proof

**Discovery/change:** In pinned b11552, 1359 copies unmasked 512x10240 F32 NextN rows twice from target to host (20 MiB each), then snapshots a fenced copy once more by CPU memcpy. The original `embd_nextn` output has public accessor/row-order contracts; skipping it without ownership proof is unsafe. Meta MIRRORED reads backend0; other split axes splice ranks. This is logical copy accounting, **not** measured PCIe or critical-path time.

**Authoritative owner(s):** QFP52 qualifies only this residual output duplication; 1359/QFP50 owns pipeline, 1348 owns deferred snapshot, QFP16/27 own scheduler **input** copies. RD71/72 fork device handoff/placement is separate; do not create `BIGCHERRY_MTP_TOPOLOGY` or another allocator/dispatcher.

**Already acting on it:** 1359 validated/default-on with historical +6..10% prefill; 1343 is superseded by merged upstream #30020. No independent QFP52/1359 work, PR or queued experiment was found in the preceding 12 hours. Excluded ongoing RAD03 mailbox/mask-only/remote-expert and RAD04 two-half prefill, and recently touched QFP47/QFP03/QFP25/PRBE115/QFP02/QFP05. Avoided BCOP191-203 and unpublished PGC15/QFP08 repeat audits.

**Unresolved action:** QFP52 Gate 0 one read-only per-batch D2H/event/host-copy/physical-rank critical-path census; only if removable exposed time can deliver >=3% E2E, prove public output and two-slot lifetimes with a host fixture, then one default-off 1359-local prototype. No new hardware lane queued.

**Terminal disposition:** `INELIGIBLE` missing physical-route/receipt; `BELOW_CEILING` if optimistic removable exposed latency <3%; `NO_SAFE_ALIAS` if output API/lifetime cannot be preserved without new global machinery; `LIFETIME_FAIL` or `NUMERIC_DRIFT` for any correctness change; `NO_EXPOSED_GAIN` if CI95-low <3% prefill/TTFT or decode/control regression >1%. Otherwise promote under 1359 after multi-request/ubatch/cancel/graph tests.

**Evidence:** pinned upstream `src/llama-context.cpp`, `ggml/src/ggml-backend-meta.cpp`, BigCherry 1359/1348 and mechanics test inspected; upstream llama.cpp #30020/#30160/#29963 and RD71/72 fork diffs checked. Local host fixture 10 assertion sites including 20 slot-wrap iterations and Python compile passed; no repository pytest, patch apply, HIP build, GPU test or new benchmark.
