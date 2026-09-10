---
id: PRBE70
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:27.196577+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# CK-001: Use Composable Kernel profiler as an oracle for hot GEMM signatures

## Description

Use Composable Kernel offline profiling as an oracle for captured hot GEMM signatures without adding CK as a broad runtime dependency.

## Steps

Select top dense/MoE signatures ranked by time*calls; run CK profiler and native MMQ/MMVQ/hipBLASLt controls with numerical parity; record throughput/resource use; if CK wins, reproduce only the useful specialization in ggml or evaluate a narrow integration.

## Detailed Solution & Technical Design

CK is an offline vendor/architecture oracle. Keep exact-signature benchmarking and provenance separate from runtime dependencies; use it to identify tile/algorithm choices BigCherry can implement or deliberately decline.

## Code Samples & Guidance



## Files

Offline CK profiling tooling, captured signature manifests, native-control comparison and evidence.

## Validation

Numerical parity; kernel throughput and resource use against native winners.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Do not add broad CK dependency; accept only a reproduced narrow specialization or explicitly documented no-port disposition backed by parity/performance evidence.

## Notes

Supersedes: RD88
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd88

## Change Log

- 2026-09-09T10:58:27.196577+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:38.618139+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.444745+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.280599+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:20:07.380935+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032047_repaired-five-more-active-succ_6361
- 2026-09-10T03:20:47.919311+00:00 (updated-by): Updated: section:ledger-events
