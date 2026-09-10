---
id: PNRO03
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:12.743676+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# Correctness-first HIP P2P transport provider for internal AllReduce

## Description

Evaluate an optional correctness-first HIP P2P transport provider for validated internal AllReduce on exactly two gfx1100 devices. Host staging remains the mandatory fallback and rejection is valid.

## Steps

1. Require validated internal AllReduce; keep P2P default OFF behind an explicit selector.
2. Gate on exactly two devices, bidirectional peer capability, peer enable, and completed correctness probes.
3. Use source-current push semantics per direction with source-owned streams/events and destination-local scratch; never use one issuer for both directions.
4. Probe deterministic nonzero asymmetric patterns both ways; disable P2P on any mismatch and fall back to host staging.
5. Exclude kernel direct peer reads/writes; this is DMA/copy-engine transport only.
6. Sweep sizes, validate every element, compare host staging/P2P, then measure real internal-AllReduce decode/prefill; reject if no stable winning envelope.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patches/1252_nro03_allreduce_p2p_provider; provider selector/probes; static tests; dual-gfx1100 evidence

## Validation

Repeated bidirectional synthetic validation, size edges, nonzero/asymmetric values, peer-enable handling, forced fallback, GPU count !=2 control, and per-direction evidence. Microbench plus real model lanes with correctness before bandwidth.

## Effort & Risk



## Standards

PGC corrected Brutus evidence; capability bits/API success are not correctness; fail closed to validated host staging.

## Acceptance Criteria

Both directed probes pass repeatedly with source-current push evidence; failures disable P2P without correctness loss; a repeatable collective-level winning envelope exists or the provider is rejected; no global default without independent topology coverage.

## Notes

Supersedes: NRO03
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro03

Supersedes: NRO03
Inherited semantic scope: preserve source-current push direction, probe/fallback, no direct peer reads, and correctness-first acceptance.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T10:52:12.743676+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:41.098537+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.061978+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.701391+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:26:09.609126+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_022800_five-nasone-successor-plans-no_4030
- 2026-09-10T02:28:00.291162+00:00 (updated-by): Updated: section:ledger-events
