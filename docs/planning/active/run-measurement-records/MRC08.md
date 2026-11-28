---
id: MRC08
order: 0
plan: run-measurement-records
state: pending
created-at: '2026-11-27T08:36:23.897113+00:00'
breadth: ''
skill: ''
created-by: agent
work: M
---

# Publish portable benchmark summaries from explicit runs

## Description

Reuse native startup evidence for explicit streaming benchmarks and export a portable Git summary automatically when configured.

## Steps

Capture the resolved startup settings once. Publish the full benchmark locally via the existing store. Generate a versioned allowlisted summary into a configured checkout, atomically and idempotently. Keep the existing PR workflow for commits and pushes.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Exercise repeat export, concurrent export, dirty checkout, failed benchmark, interrupted serving session, and host-specific sentinel values in local evidence.

## Effort & Risk



## Standards



## Acceptance Criteria

Normal serving creates only local startup evidence. Explicit benchmarks retain full local details. Configured export writes deterministic portable summaries and never emits local paths, URLs, device identifiers, or raw text.

## Notes

Draft adds opt-in --record and --export-dir, with a local record written before export and a no-replace file publish. The existing serving manifest supplies KV dtype and context. Depth matrix, warm-cache trials, expanded startup settings, export retry from a stored record and end-to-end validation remain open.

2026-11-27: actual BC 2020.23.2 release baseline: 18 TP2 rounds passed at 512/4096 tokens, 1/2/4 streams, plus real TP3 run-store publication and portable export. Fix verifies tag-bound shipped binary rather than pre-strip metadata; 32 targeted tests passed. See docs/evidence/MRC08-baseline-20261127/summary.json. TP2 raw evidence is not a sealed run record. TP3 validity remains unverified; full content model/drafter identity, matched cohort key, repeat samples, export retry and candidate acceptance integration remain required before closure.

RV4754: committed capture/export changes merged into bump and combined tests passed. Baseline must consume exact candidate and independent old runtime, matching cohort and quality; unverified data stays unverified. General export retry, depth and warm matrices after baseline.

MEN17 matched native candidate record attempt identified canonical source metadata lacks source_commit (only source_slice_id), causing run-record validation failure. Reuse worker materialization attestation and actual source verification when publishing candidate record; retain source sidecar as evidence, installed identity continues tag-bound. Legacy lab slice-only records stay unverified. 50 targeted tests passed.

## Change Log

- 2026-11-27T08:36:23.897113+00:00 (created-by): Created by agent

## Reviews

- RV4752
- 2026-11-27T08:40:12.305943+00:00 (updated-by): Updated: section:notes
- 2026-11-27T09:39:36.725327+00:00 (updated-by): Updated: section:notes
- 2026-11-27T15:14:05.801833+00:00 (updated-by): Updated: section:notes
- 2026-11-27T20:14:39.641368+00:00 (updated-by): Updated: section:notes
