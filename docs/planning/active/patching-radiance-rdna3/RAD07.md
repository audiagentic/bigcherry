---
id: RAD07
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-14T22:16:25.441629+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Per-model radiance docs: what we run, best results, what is on, what was shelved

## Description

docs/archive/radiance/LAB_HISTORY.md is a 1,144-line dated lab log mixing models, kernels and hardware; there is no per-model statement of current state. Create a clean, maintained doc set: docs/reference/engines/radiance/README.md (overview, links to parity assessment, runbook, platform design) and docs/reference/engines/radiance/models/<model>.md for qwen3.8-flash-next and qwen3.8-27b.

## Steps

1. Reconcile the Radiance overview and each per-model page to the current pinned/released engine, quantisation, hardware layout and best source-backed runs; distinguish current release facts from historical measurements.
2. Record enabled/default-off/shelved patches and settings with reasons and source links. Keep the lab history as an archive, not a second current-state document.
3. Consume RAD08's role-aware helper-layout KLD table and report reference vs measured overall/text+assistant metrics without implementing another KLD parser.
4. Verify links and update the affected model page in the same change as each new measurement or promotion.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Document link check, pin/version consistency against the current engine manifest and release records, and source/run identity verification for every displayed best/quality figure.

## Effort & Risk



## Standards



## Acceptance Criteria

1. Current Flash-Next and 27B model pages agree with the installed/pinned Radiance manifest and clearly identify historical results by source pin, run and layout.
2. Each page contains reference/best measured throughput and quality, enabled/shelved settings with rationale, and linked unfinished owner items; no unsupported claim of parity or current production support.
3. The Flash-Next page includes RAD08's actual helper-layout overall and role-specific KLD metrics when accepted, with noncomparability explicitly recorded if a required source is unavailable.
4. A link and pin-consistency check is reproducible, and each subsequent measurement or setting disposition updates its owning model page in the same change.

## Notes

Recording against model-specific docs is an owner requirement (2026-10-15).



BACKLOG TRIAGE 2026-11-04 (revalidated against main 6eb31a708): the proposed disposition "SUPERSEDED by #251/#541-#553 docs refresh" was NOT applied; this item stays open. Why: PR #251 shipped model pages, but docs/reference/engines/radiance/README.md:3,11 and models/qwen3.8-flash-next.md:4 still claim v1.14.0 is the current pin. engines/radiance/CHANGELOG.md:8 and docs/releases/notes/bc-radiance-1160.0.0.md:3 confirm the present pin is v1.16.0; the continuing-current-doc acceptance is unmet. Active items that reference this one: RAD08, RHP03. Keep RAD07 open to update the per-model and overview pages for v1.16.0 and verify their current-run claims.

CONSOLIDATION 2026-11-22: documentation/publishing owner for MEN11's outstanding helper-layout KLD comparison; RAD08 is the metrics producer and MEN20 produces component-setting A/B evidence. Do not duplicate their commands in RAD07.

## Change Log

- 2026-10-14T22:16:25.441629+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261017_012852_each-radiance-model-now-has-on_1051
- 2026-10-17T01:28:56.281155+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-04T04:23:38.248290+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4471
- 2026-11-22T09:19:47.029641+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
