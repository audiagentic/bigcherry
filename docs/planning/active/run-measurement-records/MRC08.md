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

## Change Log

- 2026-11-27T08:36:23.897113+00:00 (created-by): Created by agent

## Reviews

- RV4752
- 2026-11-27T08:40:12.305943+00:00 (updated-by): Updated: section:notes
