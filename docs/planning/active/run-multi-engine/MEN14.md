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

1. Inventory current `tools/lab/radiance/` scripts and `docs/reference/tooling/TOOL_DISPOSITION.md` against commands already available. Record absent historic wrappers separately; do not delete archived evidence or live helpers without verified replacements.
2. Make the existing build/patch/verify/promote/release command sequence the single supported Radiance change workflow. Reuse MEN20 component-setting evidence and MEN21 operator capture/replay once delivered; do not add a second launcher or evidence framework.
3. Run one genuine Radiance patch-to-release exercise and record the exact commands, pinned revision, build/evidence identifiers, validation, outcome and release in ledger/model documentation. Keep unfinished sub-gates linked to their distinct owners.

## Detailed Solution & Technical Design

Inventory of tools/lab/radiance (2026-10-19). Rule: a lab script stays only if no bigcherry command can do its job; everything that produces evidence goes through bigcherry commands.

Already managed (keep as inputs): the *-ab.json and *-kld.json files are ab-benchmark --server-config inputs; they stay as versioned configuration next to the component that owns the setting they switch.

Replaced by an existing command, retire the script: build.sh and fetch-gcc14.sh -> `bigcherry --engine radiance build --lane radiance:stock:linux-radiance` (toolchain from the platform recipe); run-radiance.sh -> `bigcherry engine-bench --engine radiance`; helper-fp16-e2e.sh, helper-fp16-kld.sh -> `build --lane` + `ab-benchmark --server-config` with the matching JSON.

No command yet (gaps, tracked): helper-evidence.sh (loop of single-setting A/Bs that stages run.json under docs/evidence/) -> MEN20 component-setting evidence command; helper-capture.sh, helper-sweep.sh, small_capture.py, heat-placement/trace_report.py, heat-placement/balance_score.py -> MEN21 moe-harness commands; fn-profile.sh, fn-profile-compare.py (rocprofv3 traces; fn-profile.sh now attaches to a native session, fn-helper.sh is deleted) -> check `profile-campaign` for radiance, else MEN21; gpu-mailbox/mailbox_probe.cpp, wire-quality/, heat-placement/heat_mock.mjs -> offline design aids, move under the component that owns the idea or retire with its plan item.

No scratch scripts outside tools/lab: untracked .capture.sh and .gemvl.sh (RAD06 one-off launchers) removed; dev-side launchers (helper-evidence-b.sh etc.) are copies of the tracked scripts and are not kept.

## Code Samples & Guidance



## Files



## Validation

Review the current lab launcher inventory against `TOOL_DISPOSITION.md`; run the affected CLI smoke tests and documentation links. Final validation is one source-backed real Radiance patch-to-release workflow receipt. A successful `bc-serve` startup alone is insufficient.

## Effort & Risk



## Standards



## Acceptance Criteria

1. Every remaining Radiance lab launcher has a recorded keep/retire disposition and, for each retirement, an existing equivalent command plus passing proof; missing historic wrappers are reported as absent, not deleted.
2. The final patch-to-release run uses maintained BigCherry commands and produces identifiable pin, patch, build, evidence, promotion and release records, with no parallel workflow introduced.
3. MEN20's component-setting A/B evidence and MEN21's operator capture/replay, where relevant, are consumed through their existing interfaces rather than reimplemented in MEN14.
4. The historic MEN11 helper-ROWS result (`men11-fire-10171108`, 8/8 clean cells, 2026-10-17) remains traceable; MEN14 does not assert new KLD parity from those throughput numbers.
5. Any remaining RAD08/RAD07 KLD/role-split and model documentation work stays assigned to those IDs, not duplicated here.

## Notes

Owner, 2026-10-15: consolidate and clean up, be process focused, follow the automated llama.cpp build/patch/test/validate/promote workflow. Performance work (RAD06 pipelining, 27B GEMM/AR) resumes on top of this.



BACKLOG TRIAGE 2026-11-04 (MERGE into RSC02; revalidated against main 6eb31a708, verdict CONFIRM): Remaining cutover (one workflow, lab launcher retired) is RSC02/RSC04 (bc-serve as the only start-up path); both exist and are open. Evidence: Remaining cutover (one workflow, lab launcher retired) is RSC02/RSC04 (bc-serve as the only start-up path); both exist and are open. Still referenced by: MEN25, MEN20. Original content kept below for history.



REOPENED 2026-11-04: an independent revalidation (against main 6eb31a708) rejected the closure recorded above, which had been based on a lighter check. It stays open. Why: MEN14.md:18-35 requires a complete build-to-release workflow, a disposition for every lab script and a final patch-to-release exercise; it explicitly identifies MEN20/MEN21 evidence and replay commands as gaps. RSC02.md:18-29 is bc-serve startup and RSC04.md:18-26 retires legacy serving configuration, not component evidence, MoE capture/replay and release workflow.

CONSOLIDATION 2026-11-22: MEN11 lab-script disposition and full workflow gate assigned here. This is not a new helper bench, quality producer or component evidence runner. The legacy fn-r9700-engine.sh/fn-bench-once.py/fn-helper-gate.sh names are absent from the current `tools/lab/radiance/` tree; explicitly verify historical replacements instead of unconditionally deleting.

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. Existing one maintained patch-to-release workflow and actual script dispositions remain here, not a new upstream programme. RSC11 removes executable manifest callers, MEN15/16/17 and RCS19/23/MEN24 provide bump sequence/qualification. Final real end-to-end run must use same maintained commands and candidate-bound evidence; preserve unique MEN20/21 evidence/capture gates, no duplicate benchmark/launcher.

## Change Log

- 2026-10-14T22:28:08.305053+00:00 (created-by): Created by agent
- 2026-10-18T13:04:03.390613+00:00 (updated-by): Updated: section:detailed_solution
- 2026-11-04T03:47:04.194775+00:00 (updated-by): Updated: section:notes
- 2026-11-04T03:47:08.148435+00:00 (state-transition): State: pending → superseded
- 2026-11-04T04:44:59.886778+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:45:04.814682+00:00 (state-transition): State: superseded → pending

## Reviews

- RV4468
- 2026-11-22T09:17:50.657888+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
- 2026-11-22T09:18:12.343518+00:00 (updated-by): Updated (no visible changes)
- RV4620
- 2026-11-25T02:12:03.269058+00:00 (updated-by): Updated: section:notes
