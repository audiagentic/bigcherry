---
id: MEN11
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-10-14T22:16:15.217144+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Radiance measurements and patch status on the campaign and evidence tooling

## Description

Radiance speed and quality runs are ad-hoc tools/lab/radiance scripts: no ab-benchmark, no evidence records, no contracts, no promotion. The helper work (libr3moe, R3_MOE_HELPER_* settings) lives in components with settings in component.toml, outside the patch lifecycle entirely. Bring radiance under the same management as llama.cpp: order-balanced A/B through the maintained harness, evidence attached to the patch or component setting it proves, and promotion/default-on decisions made the same way (lightweight tier per owner policy).

## Steps

1. Extend ab-benchmark (or engine-bench) to drive a radiance layout: engine + helper processes, graceful teardown, activation evidence (helper call counts, stage markers).
2. Quality producer: KLD against a named reference with the full metric set (RAD08).
3. Give component settings a lifecycle: each experimental setting in component.toml carries state, evidence pointer and SUMMARY like a patch package, or decide components are patches' peers in the registry.
4. Migrate the current helper evidence (rows/GEMV/heat A/Bs) into evidence records.
5. Retire lab scripts the harness replaces; keep the rest documented.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

One real A/B (e.g. R3_MOE_HELPER_ROWS 0/1) run through the harness end to end on Brutus, producing an evidence record that patch-verify-evidence (or the component equivalent) reads; offline tests for the radiance adapter.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner policy: promotion stays lightweight - reuse mechanics tests, activation evidence and a balanced A/B with identity; no heavyweight per-patch campaigns.

Correction 2026-10-15: main already has an engine-neutral activation and KL quality evidence path with a radiance adapter (#223, RAD01 PR 7) and build identity binding components (#225). Start from those: the gap is that the helper-layout runs (two processes, R3_MOE_HELPER_*) were done with lab scripts instead of through them, and component settings have no promotion path.

## Change Log

- 2026-10-14T22:16:15.217144+00:00 (created-by): Created by agent
- 2026-10-14T22:21:10.146038+00:00 (updated-by): Updated: section:notes
