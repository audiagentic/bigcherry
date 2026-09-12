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

Continue the materialized Q8_1 activation-cache work beyond closed foundation stage 1. Wire the context-owned cache into MMVQ and graph-generation lifecycle while preserving the off-mode native path. PRBE05 is the authoritative active successor to closed RD09; RD09 remains provenance only.

## Steps

1. Wire ggml_cuda_mul_mat_vec_q() through find/reserve/publish at the MMVQ materialization seam; do not alter 0200 dispatch or absorb PRBE11 paired-MMVQ fusion.
2. Begin a new cache generation at graph-evaluation entry and integrate capture_active so on-mode never grows slabs during capture.
3. Use the strengthened key: generation, view-root, exact view data address/offset, dimensions/strides and stream; use stable retained slabs, hard byte/entry caps and native fallback.
4. Add direct-Q8 producer/view-data handling, independent re-quantize plus byte-compare verification, and named hit/miss/wait/timeout/contention counters.
5. Run adversarial correctness before performance: same tensor hit, offset collision miss, shape/stride/stream miss, generation/pointer reuse miss, capacity fallback, dual-GPU isolation and direct-producer equivalence.
6. Then validate capture/replay and causal cache off/on launch and memory effects; leave dependent PRBE06 and other children separate.

## Detailed Solution & Technical Design

Context-owned bounded cache with generation invalidation and stable addresses. The cache is independent of GGML_HIP_DISPATCH_MODE. Default off is byte-for-byte native behavior; on/verify are explicit experiment arms. No synchronous verification or slab growth during capture. Cache identity must include the exact view data address/offset; storage must remain in non-relocating slabs. Stage 1 foundation is closed; this item implements stage 2 caller and graph-entry wiring.

## Code Samples & Guidance



## Files

patches/1235_rd09_q81_activation_cache_foundation; ggml-cuda MMVQ materialization and graph-entry seams; hip-q81-cache API; adversarial key/capture fixtures; independent re-quantize/byte-compare verifier; rd09-only campaign evidence.

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

Supersedes: RD09 (closed historical predecessor). PRBE05 is the actionable owner. PRBE06 and PRBE18 consume PRBE05; PRBE11 consumes it only if a source audit proves a real dependency. Preserve source identities ff6fde5046ffb86672e05da640d2bfb20d4bfdfc and rebased 299f6e985... in provenance; do not make RD09 a live dependency target.

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
- 2026-09-12T09:52:10.845595+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:notes
- chg_20260912_095506_cleaned-the-active-rdna-boost_4906
- 2026-09-12T09:55:06.200665+00:00 (updated-by): Updated: section:ledger-events
