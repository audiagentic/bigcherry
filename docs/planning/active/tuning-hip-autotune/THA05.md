---
id: THA05
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:48:25.987562+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# bigcherry tune-campaign: single-command record->tune->correctness->promote->replay orchestrator

## Description

Deliver the production single-command tune-campaign orchestrator for record→tune→correctness→promote→replay with deterministic receipts and configurable runtime profiles.

## Steps

Implement workflow stages and thin CLI; reuse server lifecycle helpers; expose source_root; persist per-stage build/source/artifact identities and effective context; use named production profiles with bounded tune context and explicit preflight; replay from tune build manifest; reproduce the validated dual-XTX Qwen3.8-27B baseline and coverage behavior without legacy shared-checkout mutation.

## Detailed Solution & Technical Design

Production orchestrator, not smoke campaign reuse. Fail closed on unsafe VRAM/context and preserve campaign APIs. Receipt must include run/source/build IDs, manifest, cache, promoted count and profile.

## Code Samples & Guidance



## Files

tools/bigcherry/tuning/workflow.py; cli/tuning.py/main.py; CampaignLaneResult; receipt/build JSON; runtime profiles; release persistence idempotency.

## Validation

Single-command real Brutus record/tune/correctness/promote/replay, receipt completeness, 20/39-style promotion baseline where reproducible, bounded context and replay coverage.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

One deterministic command completes all stages with immutable receipt/provenance, no shared-checkout interference, explicit effective context, and no regression against validated dual-XTX baseline.

## Notes

Supersedes: HI130
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi130

## Change Log

- 2026-09-09T10:48:25.987562+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:28.943683+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.810110+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.293707+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:26:59.894624+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032737_repaired-four-major-tuning-suc_7897
- 2026-09-10T03:27:37.881650+00:00 (updated-by): Updated: section:ledger-events
