---
id: BCOP46
order: 46
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-07T10:04:00+11:00'
created-by: agent
priority: P1
---

# PRBE58 MTP hidden-state handoff disposition

## Audit result

PRBE58 is the authoritative owner for embedded-MTP hidden-state handoff traffic. PRBE57 owns NextN placement and PNRO10/1261 owns ctx_other scheduler-backend coverage for genuinely different draft-model device lists. Upstream llama.cpp #26636 confirms the latter boundary; PNRO10 already records that its path does not activate for embedded MTP tensor split.

The old generic alias/P2P/host-bounce design is narrowed to a measurement-first gate. Production hardware lacks normal GPU P2P, and current MTP multi-ubatch evidence requires explicit lifetime/order correctness before any borrowed device-resident handoff.

## Unresolved action

Instrument the existing target-hidden-state -> MTP input boundary and measure copies, bytes, host staging and waits for MTP depth 1/3/7 at shallow/~80K/>=160K context on gfx1100/gfx1201, including multi-ubatch, single-GPU and MTP-off controls.

## Terminal disposition

- **<3% MTP decode wall:** close the copy-reduction sub-slice; do not port 41a8ca78.
- **>=3%, same backend:** prototype only a lifetime-fenced borrowed device-resident view.
- **>=3%, different backend:** wait for PRBE57 measured placement, then reuse existing backend-copy/persistent staging; no private P2P transport.
- Promote only with correctness/work accounting intact, CI95-low-positive >=3% E2E MTP decode gain on a production architecture, <=1% cross-architecture regression and <=1% prompt regression.

## Dependencies / blockers

PRBE57 placement must precede any cross-device implementation choice. FMTP03's currently active scheduler-lifetime work is protected by the 12-hour rule and was not modified. No hardware result is claimed by this audit.
