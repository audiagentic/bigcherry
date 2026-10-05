---
id: BCOP36
order: 36
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T21:50:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: S
---

# Track whole-system speculative decoding qualification

## Description

Action/disposition ledger for treating MTP/draft speculation as an end-to-end pipeline. `patching-flash-next-mtp-pipeline` (FMTP) remains the technical owner for MTP/speculation mechanisms. RPL01 owns cross-capability/device placement comparisons. BCOP36 must not create another speculation controller.

## Actions

1. Extend the relevant FMTP item(s), not BCOP36, with common end-to-end evidence: proposed/accepted tokens, acceptance by depth, draft/verify time, rollback/recompute, KV/memory cost and effective accepted tokens/s.
2. Use existing FMTP mocks/controls to compare MTP-off and supported draft depths before proposing adaptive policy.
3. Test same-device first. Split-device draft/target is considered only when RPL01 plus measured transfer/sync costs predict a real opportunity.
4. Require short/long-context and multi-request integrity; raw draft throughput is not a promotion metric.
5. Runtime adaptation remains blocked until an offline selector predicts held-out winners and transition cost is bounded.
6. Record terminal disposition: `static-config-sufficient`, `FMTP-extension`, `placement-gated`, or `rejected-overhead`.

## Gate

Promotion requires >=5% effective end-to-end TG with <=2% regression in unaffected workloads and full correctness. FMTP owns implementation; RPL01 owns competing VRAM/device-use comparison.

## Related

FMTP02-07; RPL01; BCOP30/32; existing sparse-FA/KV owners.
