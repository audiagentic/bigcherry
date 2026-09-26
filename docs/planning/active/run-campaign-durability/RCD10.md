---
id: RCD10
order: 10
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:23.945064+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Hardware acceptance, consolidation, and retirement of the lab queue scripts

## Description

Accept the job service on Brutus with four jobs (gfx1100 ordinary; gfx1201 alternate ROCm; gfx1030 alternate model; dual gfx1100) and forced incidents (harness failure + same-run retry, client disconnect, disable/re-enable, disk guard, promotion between sessions). Consolidate existing components per RCD02 (core/tree_activity leases, experiment/bundle.run_managed + telemetry.py, tuning/journal.py, build/resource locks; fold RCD01's durable-state protocol in rather than keeping a separate design). Then retire tools/lab/plan-qualification queue.sh, run_campaign.sh, make-serial-2.sh, work-root.sh, switch scripts and watcher loops (TOOL_DISPOSITION -> ARCHIVE).

## Steps

1. Acceptance run + evidence.
2. Consolidation changes.
3. Migrate in-flight batches; archive lab scripts.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Acceptance record; no remaining references to the lab queue in docs/skills.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-26T00:52:23.945064+00:00 (created-by): Created by agent
