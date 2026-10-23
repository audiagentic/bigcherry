---
id: MET13
order: 0
plan: patching-moe-expert-tiering
state: superseded
created-at: '2026-10-12T12:21:43.616815+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# MET13 — superseded duplicate of MET12 host-expert prefetch

## Description

**Disposition (2026-10-15): `SUPERSEDED_DUPLICATE`.** The requested Strata-style layer-ahead host-expert staging, dedicated copy stream and event-protected ring are the same capability already owned by [MET12](../../active/patching-moe-expert-tiering/MET12.md), created 2026-10-08. No MET13 implementation, independent scheduler, ring, allocator, cache or hardware experiment is authorised. The earlier proposed MET13 steps are superseded; their text remains in Git history.

**Source reconciliation:** pinned b11552 already implements exact post-router selective host-weight uploads in `llama_context::sched_copy_experts` through the native scheduler callback; 1336 is superseded. Strata's stream-all-ahead mode is a distinct high-byte option, not proof that the same approach is beneficial for BigCherry's fully resident three-GPU layout. See MET12's BCOP216 audit for exact paths, coverage/PCIe bounds, correctness gates and terminal decisions.

**Owner:** MET12 (conditional pre-router staging); MET06 (host-weight scheduler/pipeline); MET07 (cache/profile); MET10 (host-tail graph). Active Radiance helper and prefill development is untouched.

**Validation:** 10/10 disposable host-only admission/lifetime tests passed under MET12; no GPU run, benchmark, repository test, or implementation change. Closed by BCOP216.

## History
- 2026-10-12: Created with overlapping prefetch/ring proposal.
- 2026-10-15: Retired as duplicate; technical work consolidated in MET12.

## Steps


## Detailed Solution & Technical Design


## Code Samples & Guidance


## Files


## Validation


## Effort & Risk


## Standards


## Acceptance Criteria


## Notes


## Change Log

- 2026-10-23: state 'done' (invalid) corrected to superseded
