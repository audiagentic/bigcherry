---
id: BCOP195
order: 195
plan: patching-bc-optimizations
state: done
created-at: '2026-10-14T00:05:00Z'
created-by: agent
priority: P2
work: S
---

# PRBE114: retire unqualified tensor-split SSM_CONV Meta rewrite

**Discovery/change:** Current 1263 already gates channels-major on `model.split_mode() != TENSOR`, retaining the stock time-major RESHAPE/TRANSPOSE/CONCAT and a `declined_split` marker. The b11233 Meta `nr[0]==1` startup crash predates this guard. Pinned b11552 `handle_reshape` retains the invariant; 1263's SSM_CONV axis mapping does not fix an earlier reshape mismatch. PRBE114's pending generic Meta rewrite and 1263 TESTING's positive-marker requirement for tensor split are stale.

**Owners/consolidation:** PRBE41/1263 owns the existing single-device channels-major implementation and contract; PRBE114 is terminal for its speculative tensor-split extension. No new Meta scheduler/cache/allocator/dispatch or duplicate plan. No independent PRBE114/PRBE41/1263 activity in the preceding 12 hours; active Radiance mailbox/R9700 expert-helper, RAD01 release and their hardware lanes remain protected. BCOP194 was not present on this branch; this item uses BCOP195 to avoid collision with a previously prepared but unpublished BCOP194 proposal.

**Evidence:** Historical validated single-device `tierA-qwen4b-q6k` pp512: gfx1100 about +1.13%..+1.43%, gfx1201 +2.37%/+2.58%, full-vocab parity; old dual-XTX `-sm tensor` Qwen3.8-27B Q8_0 pp4096 +0.14% [0.03%,0.24%] is a fallback route, not a channels-major speedup. Upstream llama.cpp #23843 fixes a different tensor granularity problem; AMD-Ecosystem #52 reports external gfx1151 single-GPU results. No new GPU gain.

**Disposition/gate:** Close PRBE114 without implementation. Reopen only if current-pin, physical-device-bound tensor-split traces show >=3% exposed E2E transpose-removal opportunity; then require first-failing-node graph fixture, exact split-state shape/ratio proof, safe fallback, recurrent-state/MTP parity and paired gfx1100/gfx1201 qualification. Otherwise no experiment queue or patch promotion. Host/source checks 16/16 passed; no HIP compile or GPU run this audit.
