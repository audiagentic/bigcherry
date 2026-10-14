---
id: MEN14
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-10-14T22:28:08.305053+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P0
work: L
---

# Radiance consolidation: one engine workflow from build to promote, lab scripts retired

## Description

Umbrella for aligning radiance with the automated llama.cpp workflow: build -> patch -> test -> validate -> promote -> release, every step a bigcherry command with recorded evidence. Sequence: RAD09 (reconcile branch, merge to main) -> MEN10 (engine-aware pin/bump/rebase) -> MEN11 (radiance runs through ab-benchmark and the activation/KL evidence path from #223; component settings get a promotion path) -> RAD08 (KLD metric set) -> RAD07 (per-model docs) -> MEN13 (release line) -> MEN12 (first real radiance bump + release as the end-to-end fire test).

## Steps

1. Inventory tools/lab/radiance: for each script, the bigcherry command that replaces it, or why it stays a lab tool.
2. Land RAD09; shelve-to-main any branch-only work (few-rows kernel already default off, evaluated).
3. Execute the sequence above; each step closes only with its tool used on a real radiance run.
4. Final gate: one radiance change taken from patch edit to a bc-radiance release using only the documented workflow.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

The final gate run, recorded in the ledger and on the model page.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner, 2026-10-15: consolidate and clean up, be process focused, follow the automated llama.cpp build/patch/test/validate/promote workflow. Performance work (RAD06 pipelining, 27B GEMM/AR) resumes on top of this.

## Change Log

- 2026-10-14T22:28:08.305053+00:00 (created-by): Created by agent
