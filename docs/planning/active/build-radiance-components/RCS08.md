---
id: RCS08
order: 8
plan: build-radiance-components
state: pending
created-at: '2026-10-18T12:57:58.871719+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# Component lint: size budget, ownership, and evidence tied to status

## Description

Keep the split from regressing. A component lint (run with patch-lint) checks: every file under a component is owned by exactly one component.toml; every setting read by code appears in exactly one component.toml; status production or validated requires an evidence run.json that passes the same checks as patch evidence (arms differ in that setting alone, executed activation record, no failed cell); a source-file size budget (warn above 800 lines, fail above 1,200, with a recorded allowlist while RCS02-RCS03 are open).

## Steps

1. Add the lint beside patch-lint, reusing the activation/evidence verification code.
2. Wire it into the pin-bump gate and CI.
3. Tests with fixture components for each failure.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Lint fails on a fixture with an unowned file, an undeclared setting, a validated setting without evidence, and an oversize file; passes on the real tree.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Can start before the moves. Also covers the earlier idea of tying production/validated status to evidence.



BACKLOG TRIAGE 2026-11-04 absorbed items (superseded into this one; their notes keep the full history): MEN20: RCS08 (component status/evidence CLI) exists and is open; no duplicate command.



CORRECTION 2026-11-04: the earlier note that MEN20 were absorbed into this item is withdrawn. An independent revalidation found each either still has open scope of its own (reopened) or was delivered outright. This item's scope is unchanged.

## Change Log

- 2026-10-18T12:57:58.871719+00:00 (created-by): Created by agent
- 2026-11-04T03:48:03.724548+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:45:56.048626+00:00 (updated-by): Updated: section:notes
