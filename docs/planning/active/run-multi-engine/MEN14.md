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

Inventory of tools/lab/radiance (2026-10-19). Rule: a lab script stays only if no bigcherry command can do its job; everything that produces evidence goes through bigcherry commands.

Already managed (keep as inputs): the *-ab.json and *-kld.json files are ab-benchmark --server-config inputs; they stay as versioned configuration next to the component that owns the setting they switch.

Replaced by an existing command, retire the script: build.sh and fetch-gcc14.sh -> `bigcherry --engine radiance build --lane radiance:stock:linux-radiance` (toolchain from the platform recipe); run-radiance.sh -> `bigcherry engine-bench --engine radiance`; helper-fp16-e2e.sh, helper-fp16-kld.sh -> `build --lane` + `ab-benchmark --server-config` with the matching JSON.

No command yet (gaps, tracked): helper-evidence.sh (loop of single-setting A/Bs that stages run.json under docs/evidence/) -> MEN20 component-setting evidence command; helper-capture.sh, helper-sweep.sh, small_capture.py, heat-placement/trace_report.py, heat-placement/balance_score.py -> MEN21 moe-harness commands; fn-helper.sh, fn-profile.sh, fn-profile-compare.py (rocprofv3 traces) -> check `profile-campaign` for radiance, else MEN21; gpu-mailbox/mailbox_probe.cpp, wire-quality/, heat-placement/heat_mock.mjs -> offline design aids, move under the component that owns the idea or retire with its plan item.

No scratch scripts outside tools/lab: untracked .capture.sh and .gemvl.sh (RAD06 one-off launchers) removed; dev-side launchers (helper-evidence-b.sh etc.) are copies of the tracked scripts and are not kept.

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
- 2026-10-18T13:04:03.390613+00:00 (updated-by): Updated: section:detailed_solution
