---
id: THA01
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:48:09.536774+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Emit intrinsically-complete nested provenance from the C++ runtime transform writer (retire the flat adapter)

## Description

Complete and hardware-validate nested runtime transform provenance emission, then retire the transitional flat adapter only in a separately gated follow-up.

## Steps

Verify runtime build descriptor hash matches offline manifest; update hip-autotune-tuner.cu flush to nested runtime schema with schema version, hardware architecture, build descriptor hash and evidence references; add nested fixture ingestion/analyze_gap provenance_complete tests; compile/run real Brutus tuning and ingest artifact; keep flat adapter until real nested artifacts exist, then remove in separate item with fail-closed old-header rejection.

## Detailed Solution & Technical Design

Make runtime transform records intrinsically complete so DB identity resolution is unnecessary. Preserve current flat adapter as transitional and never alter tuning timing/behavior. Use the same manifest inputs and hash algorithm as offline cache; bind by source revision, manifest hash and build descriptor hash.

## Code Samples & Guidance



## Files

vendor/llama.cpp/ggml/src/ggml-cuda/hip-autotune-tuner.cu; transform_records.py; transform_loader.py; tuning tests; Brutus artifact.

## Validation

Hash parity fixture; nested runtime-shaped ingest/analyze with provenance_complete=True; full offline suite; real nested Brutus artifact ingest without DB identity resolution.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Complete only after C++ writer emits valid nested records, offline and real hardware artifacts ingest with complete provenance, and flat-adapter retirement is separately planned and gated.

## Notes

Supersedes: HI104
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi104

## Change Log

- 2026-09-09T10:48:09.536774+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:09.550721+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.789339+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.264375+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:24:37.131528+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032455_repaired-two-tuning-successors_8434
- 2026-09-10T03:24:55.035392+00:00 (updated-by): Updated: section:ledger-events
