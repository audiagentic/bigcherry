---
id: BCOP47
order: 47
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-07T14:04:00+11:00'
created-by: agent
priority: P1
---

# PRBE67 HIP graph staleness disposition

## Audit result

PRBE67 is the authoritative owner for the same-UID HIP graph / Flash Attention staleness hypothesis. Current upstream still returns early from graph property checking when a non-zero cgraph UID matches the cached graph UID. Newer warmup/reset handling does not remove that early return.

PRBE66 overlaps the same mechanism and should not produce a second implementation. Its graph-disable concept is retained only as a qualification control or rollback.

## Unresolved action

Instrument the existing UID early-return seam and reproduce the gfx1201 long-context failure. Record UID, node count and a compact FA K-tensor signature. Compare graphs-on and graphs-off.

## Terminal disposition

- Same UID with no changed FA signature at failure: reject this hypothesis and remove diagnostics.
- Same UID plus changed FA signature: prototype only a guard that falls through to the existing full node-property comparison.
- Do not add another graph cache, fingerprint table, scheduler or allocator.
- Promote with graphs-off output parity, zero stale replays, stable-shape no-extra-recapture proof, <=1% stable gfx1100/gfx1201 regression, and repeated same-process/multi-ubatch/long-context coverage.

## Dependencies / blockers

Upstream issue #23579 is independent evidence that gfx1201 multi-GPU FA can fail under HIP graph capture and that graphs-off is a useful control; it does not establish the UID mechanism. No hardware test or prototype was run by this audit.
