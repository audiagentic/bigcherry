---
id: PRBE05
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:53:47.797556+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Implement per-graph Q8_1 activation cache foundation

## Description

Continue the materialized Q8_1 activation-cache work beyond closed foundation stage 1. Wire the context-owned cache into MMVQ and graph-generation lifecycle while preserving the off-mode native path.

## Steps

- Wire ggml_cuda_mul_mat_vec_q() through find/reserve/publish at the MMVQ materialization seam; do not alter 0200 dispatch or absorb RD12 fusion.
- Begin a new cache generation at graph-evaluation entry and integrate capture_active so on-mode never grows slabs during capture.
- Use the strengthened key: generation, view-root, exact view data address/offset, dimensions/strides and stream; use stable retained slabs, hard byte/entry caps and native fallback.
- Run adversarial correctness before performance: same tensor hit, offset collision miss, shape/stride/stream miss, generation/pointer reuse miss, capacity fallback, dual-GPU isolation and direct-producer equivalence.
- Then validate capture/replay and causal cache off/on launch and memory effects; leave dependent PRBE06 and other children separate.

## Detailed Solution & Technical Design

Context-owned bounded cache with generation invalidation and stable addresses. The cache is independent of GGML_HIP_DISPATCH_MODE. Default off is byte-for-byte native behavior; on/verify are explicit experiment arms. No synchronous verification or slab growth during capture.

## Code Samples & Guidance



## Files

patches/1235_rd09_q81_activation_cache_foundation; ggml-cuda MMVQ materialization and graph-entry seams; hip-q81-cache API; adversarial key/capture fixtures; rd09-only campaign evidence.

## Validation

Full key matrix including same-root different-offset MUST miss, pointer reuse, dimension/stride/stream/generation misses, capacity/native fallback, dual-GPU isolation, graph warm-up/capture/replay and stable addresses; Q8 byte identity; cache counters; off-vs-on causal launch/memory/perf evidence.

## Effort & Risk



## Standards

Bounded cache; generation safety; no stale pointers; exact quantizer reference; no graph-time allocation; independent of dispatch mode.

## Acceptance Criteria

Cache integration passes all key and capture gates with zero Q8 block mismatches; off mode is unchanged; exhaustion and nonqualifying paths fall back natively; positive performance/launch evidence is required before promotion and dependent children remain separate.

## Notes

Supersedes: RD09
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd09

## Change Log

- 2026-09-09T10:53:47.797556+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:10:28.989477+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events



- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.147212+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.822900+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:31:39.715349+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023232_the-next-three-rdna-successors_5807
- 2026-09-10T02:32:32.971906+00:00 (updated-by): Updated: section:ledger-events
- chg_20260911_221341_documented-the-real-status-of_8365
- 2026-09-11T22:13:41.602225+00:00 (updated-by): Updated: section:ledger-events
