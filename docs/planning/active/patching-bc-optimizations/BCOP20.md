---
id: BCOP20
order: 20
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T04:49:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Remove context-wide synchronization from target-to-drafter MTP handoff

## Description

Backfill of the earlier QFP08 audit. Existing traces identified repeated full-context synchronization around MTP hidden-state handoff. Before adding broader speculative scheduler complexity, qualify a generation-tagged event-driven handoff with persistent double buffers.

## Steps

1. Reconfirm current synchronization counts and blocked wall time on the live branch; close this item if later code already removed the dominant waits.
2. If still present, use generation-tagged double buffers and backend events for target-hidden-state -> drafter transfer.
3. Prefer direct peer transfer only if qualified; otherwise use async D2H/H2D with persistent pinned buffers.
4. Preserve producer/consumer lifetime and rejection-safe recurrent/KV state.
5. Do not add cross-round lookahead until the handoff primitive itself is correct and profitable.

## Related

QFP08, FMTP03, QFP27.

## Acceptance Criteria

- Dominant handoff synchronization classes are reduced by >=50% blocked time.
- End-to-end decode improves >=5% on one representative lane and >=3% on another, with no lane >2% slower.
- Acceptance and target correctness remain unchanged.
- No duplicate generic scheduler is introduced.
