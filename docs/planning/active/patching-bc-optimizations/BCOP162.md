---
id: BCOP162
order: 162
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-12T12:35:35+00:00'
created-by: agent
priority: P3
work: S
---

# QFP02: fence residual k-pool tail scan to a measured position-mode route

**Discovery:** Validated 1292 already removes the O(context) rebuild for Qwen4Exp MTP tail edits (historical ~14% lower step latency at 80K). Its by-order path has no additional demonstrated optimisation. The shared position-aligned path may re-scan an unchanged sparse prefix when no complete pool survives, because the cut resets `sq.j_next` to zero. This is source-derived; not a reproduced serving regression.

**Owners / consolidation:** QFP02 owns the existing 1292 technical mechanism and conditional position-mode bound; QFN01 owns the operational Flash-Next profile and QFP18 the historical promotion. No new patch, cache, scheduler, allocator, selector, telemetry system or GPU lane. Keep QFP03/QFP04/QFP17/QFP53 and active Radiance/QFP15/release work untouched. Existing 1292 `SUMMARY.md` now names technical and operational owners.

**Work already done:** Patch 1292 was validated/promoted in QFP18 and re-anchored at b11474 on 2026-10-07; no independent 1292/QFP02 commits, PRs or queued tests in the preceding 12 hours. A standalone 70,000-case host model in this audit found zero pool-list mismatches; synthetic sparse 80K candidate scans fell from 79,985 to 10 with a conditional bound. No build or hardware benchmark was run.

**Unresolved action / terminal gate:** First prove a real non-order hybrid-indexer tail-edit route and >=5% non-overlapped step time in this scan, with >=3% Amdahl E2E ceiling. If absent, close without implementation. If present, extend only 1292's `bc_tail_cut` position-mode `j_next` bound and require forced-full-rebuild parity across edits, sharing, snapshots and generations, followed by isolated ABBA and no-regression controls. Any mismatch, absent route, CI95-low <3% E2E gain or >1% control regression rejects the extension; validated 1292 remains unchanged. Dependencies and exact tests are in QFP02.
