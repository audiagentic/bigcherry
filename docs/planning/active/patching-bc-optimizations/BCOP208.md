---
id: BCOP208
order: 208
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-14T11:08:30Z'
created-by: agent
priority: P2
work: S
---

# QFP11: correlate AR events before boundary optimisation

QFP11's ar-boundary.py pairs ranks by ordinal. QFP09's ar-segment.py independently chooses nearest events within 2 ms without one-to-one matching. Missing/extra events can fabricate skew and duplicate remote matches. Keep historical 78/82 us skew provisional until correlated trace replay. QFP09 owns one shared matcher; QFP11 owns the boundary decision; QFP01/1291 owns epoch instrumentation. Do not modify active RAD02/RAD03/RAD04 work.

Gate: five synthetic host controls passed; real rocprof replay, HIP and GPU tests not run. Require epoch/dispatch identity, all trace shards, one-to-one monotonic matching, zero ambiguity, >=99.5% coverage, clock/stream proof. If ambiguous, terminal UNVERIFIED_TRACE. If <3% exposed E2E, NO_EXPOSED_BOUNDARY_GAIN. Otherwise hand off to existing rank-skew, collective or submit owners. Upstream #30453 already covered by 1007; #30329 adds realloc counters only after pin update.
