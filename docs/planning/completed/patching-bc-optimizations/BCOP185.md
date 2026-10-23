---
id: BCOP185
order: 185
plan: patching-bc-optimizations
state: completed
created-at: '2026-10-13T12:05:00Z'
created-by: agent
priority: P2
work: S
---

# PRBE67: graph UID reuse and binding-identity evidence gate

## Description

**Finding:** Pinned llama.cpp b11552 and upstream master skip tensor/source-property comparison when nonzero graph UIDs match. This is a possible stale replay admission, not a reproduced AMD failure. The existing slow comparison is sufficient; a second graph fingerprint is unnecessary. The evaluated 1356 patch's normalized trace digest can miss a changed absolute buffer base. Neither historical 1356 UID-bypass nor locking attempt resolved its 98K output-identity failures.

**Ownership:** PRBE67 owns the bounded UID-reuse diagnostic. QFP41/1356 retains worker-mode race ownership; MSM02/1340 owns arena binding; PRBE66 is historical duplicate planning. No new scheduler, cache, graph map, allocator or GPU queue. Recently active Radiance/RAD01 paths remain untouched.

**Gate:** On an idle lane, correlate a same-UID absolute pointer/shape/stride mismatch with a graphs-on correctness failure before changing runtime code. If absent, close UID_HYPOTHESIS_NOT_CONFIRMED. If causal, test one default-off fall-through to the native property comparison, with 12 repeated runs, graphs-off logits/KLD/greedy parity, multi-request and ubatch safety, no skipped work, and at most 1% stable decode regression.

**Validation:** 13/13 source assertions, 10/10 synthetic host tests and Python compilation. No HIP build, GPU run, repository pytest or new throughput result. Upstream llama.cpp PR #29287 is an RPC-only draft; SGLang issue #34384 is an external graph-slot compatibility precedent.

## Steps


## Detailed Solution & Technical Design


## Code Samples & Guidance


## Files


## Validation



Source-static and host-only checks recorded in this item's disposition/evidence text; no hardware run was required for this closure.

## Effort & Risk


## Standards


## Acceptance Criteria



The disposition and ownership decisions described above are recorded; no further implementation is owed by this item.

## Notes


## Change Log

- 2026-10-23: state 'done' (invalid) corrected to completed
