---
id: MEN24
order: 24
plan: run-multi-engine
state: pending
created-at: '2026-10-21T00:30:00+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Faster, less fragile engine bumps: composition, native launch and overlay integration order

## Description

The radiance 1.9.0 bump (2026-10-20) took about an hour, but measured compile time is about 6 minutes on dev (20 cores): engine about 75 s wall, libr3 about 4.7 min (one unit, r4d_gemm_mxfp4a8_decode, alone 264 s and the critical path), moe-remote 10 s, ngram-helper 2 s. The rest is pipeline: the engine is built twice (compile gate, then the lane build for the CPU smoke), there is no compiler cache, the patch-test step runs the whole offline patch suite (208 files, 1,444 tests, about 140 s, only 2 files about radiance), and selftest/bench binaries are built in the gates. The bump also stopped on 10 libr3 native rewrite blocks that upstream had moved: in-source text rewrites of libr4d break on every release, while our own components (moe-remote, ngram-helper) needed only the ABI rebuild.

## Steps

1. Record per-phase wall time in the pin-bump run record (the 1.9.0 run's timestamped log is the first sample).
2. Build once: the lane-build phase reuses the compile gate's build when source, options and targets are identical, instead of building again.
3. Compiler cache: ccache as CMAKE_C/CXX/HIP_COMPILER_LAUNCHER for lane and bump builds, cache directory on the workspace volume; record hit rate.
4. Engine-scoped bump tests: run_patch_tests runs only the bumped engine's patch and component tests (a radiance bump does not run llama.cpp patch tests), plus the component ctest targets.
5. Gates skip selftest and benchmark targets (r4d_selftest 51 s); they stay in the full lane build.
6. Split or reorder the slowest libr3 unit so it is not the build's tail (build it first, or split mxfp4 decode by shape).
7. Nightly drift check: run the native-rewrite drift and patch-rebase checks against upstream main without a GPU, so anchors that break are seen when upstream changes them, not on bump day.
8. Fewer in-source rewrites: move self-contained enhancements out of libr3/native/*.rw into kernel files of our own with their own rows (candidates: the all-reduce timing hook and 4-bit wire, the pipelined norm), leaving rewrites only for what must change inside upstream code. New enhancements (RAD15) are written as rows from the start.
9. Keep the gate suite green: failing tests on main are fixed or precisely skipped, never waved through as pre-existing.

## Detailed Solution & Technical Design

Order A–H (one programme checklist, no new plan item): A exact candidate and current-pin self-triage, RCS23 origin/row/dependency and native interface impact; B MEN16/PA49 declare/pull/audit and pristine coupling; C RCS19 exact recipe patch materialization then authoritative rewrite/ownership checks; D RCS19/20/21 affected host/device compile and link/export, actual transitive closure or conservative full set; E MEN16 canonical lane build/reuse plus RSC01/02/05/06 and RSP01 matched native parser, all YAMLs and model/row roster; F MEN15 native CPU smoke and separate card-free owner tests; G MEN17 exact candidate release acceptance (RCS25 discrete affected-unit evidence from RCS23 is obtained before the full release build), RSC11 lifecycle and PWR10 qualified power restoration; H reaudit, bound evidence, explicit upstream adoption disposition and RSP01 deliberate release. Existing pin-bump phases and adapter interface are sole orchestrator; exact phase names may be adapted to existing state machine. RCS15 provider decision gates large gfx11 overlay changes, not early gate work. RSC11 S18 native launch and PWR08 converge in one cutover, no interim manifest power producer. RCS19 gate integration, RCS25 harness, MEN15/16/17 binding and RSC11 can run in parallel; no separate programme. Nightly read-only exact-ref triage with changed dependency/row/native boundary report, no repin or qualification.

Refine order: developer A–D composition and affected syntax/compile/link then RCS25 discrete one-card resource/byte/replay WITHOUT full engine or full plugin. Release A–H remains exact selected composition and affected evidence -> one integrated lane -> matched native inspector/roster and actual native CPU smoke -> required Gate E installed canaries and affected dense/helper/quality/power cohorts -> reaudit and deliberate release. First native S18 migration full supported deployment matrix; routine releases two existing Gate E production layouts plus affected paths and required independent gates, not every historical experiment. No fixed roster count. RCS15 compare same-source generated/native/reference on observed candidate when authorized; do not wait for all optional RCS22 family work for a useful bump. Freeze actual target before porting; no main-head release assumption.

## Code Samples & Guidance



## Files



## Validation

The next radiance bump records per-phase times; target under 15 minutes end to end when no rewrites drift; one engine build per bump; a radiance bump runs no llama.cpp tests; the nightly drift job reports on upstream main.

## Effort & Risk



## Standards



## Acceptance Criteria

Retain all earlier speed and correctness gates; prove exact staged order, no required skipped/unverified proof accepted, phase result identities and invalidation, same selected recipe and candidate build. Record per-phase timing, compiler hits, affected/full-fallback unit counts, duplicate build avoidance and human reconciliation time; compare at least two real candidate bumps through RCS23, including resume/rollback and qualified release receipt. Earlier under-15-minute target is for clean software path, not GPU queue/qualification guarantee. Report actual timings, no claim from card-free alone. Cross-engine unchanged; no new driver.

Record real prior and candidate commits/tag vs main, changed paths, affected/dependent rows and adoption classification, candidate-specific disposition. Preserve simple Git/version fields and existing native ABI/layout tests. Two real bump evidence runs retain timing and distinct discrete/full release qualification; no claim source overlap means failure. Reuse unchanged compatible proofs only. No skipped required gate counts as pass.

### 2.3.0 fast bump implementation (owner direction 2026-11-27)

The [fast-gate review](../../../evidence/2026-11-27-radiance-230-bump/FAST-GATES.md) controls test scope. A-D: existing affected patch/compose, host/device compile and smallest real owner harness; no full engine build per repair. Today ComponentGate.build already builds engine+components and lane-build repeats; MEN16/24 reuse exact same canonical output for full build, plugin ABI and declared CTest at final release checkpoint; until wired, scoped checks are developer evidence, not formal gate completion. Unknown scope widens compile only to affected closure/component, not all GPU tests. Use prior proof only with verified compatible contracts. Run no shared-KV L/S/C redesign (RSC16-18) or IQ6R conversion (XDP11) as bump prerequisites. Keep rad0062 untested outside production. Existing legacy capacity/admission and current models are requirements, not optional experiments. One installed candidate acceptance cohort for each distinct affected capability, reuse existing receipts and combine tests in the same load. No new tool or test framework.

### 2.3.0 migration gate (owner direction 2026-11-27)

Use the exact candidate ledger in docs/evidence/2026-11-27-radiance-230-bump/MODEL-COMPATIBILITY.md. Before composition pruning, freeze the approved YAML/model roster from RSC12 and actual container planes, draft forms and old selected graph/provider; unknown scope blocks retirement. Stage B must resolve native ABI44 vs composed local ABI (proposed 45), not just edit coupling numbers. Stages C/D preserve all required rows, including helper and draft paths. At E run matched native explain and declaration/selection for every approved YAML; parser success alone is not graph coverage. At G this bump is a broad affected-path migration, not just two routine canaries: MEN17 proves each distinct required capability and all four model containers on candidate hardware. Reuse one measured run for equivalent variants only with documented semantic equivalence; no exhaustive Cartesian GPU sweep. At H reject a missing model, unqualified provider, silent fallback, or unapproved replacement. Do not delete a YAML to make coverage pass. Keep prior release and containers for rollback. No duplicate orchestrator or new roster.

## Notes

Measured 2026-10-20 from the .ninja_log files of lane build 4babe498 and a local run of tools/tests/patch. Related: MEN10-MEN19 (managed engine workflow), RAD15 (engine rows instead of rewrites).



BACKLOG TRIAGE 2026-11-04 absorbed items (superseded into this one; their notes keep the full history): MEN19: pin-bump has the drift phase (NATIVE_REWRITE_DRIFT) and caught the v1.16 rewrite break; remaining pin-release confirmation work is MEN24.



CORRECTION 2026-11-04: the earlier note that MEN19 were absorbed into this item is withdrawn. An independent revalidation found each either still has open scope of its own (reopened) or was delivered outright. This item's scope is unchanged.

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. Existing 1.9.0 timing and compiler-cache goals retained as historical evidence; not a measured current result. RCS19 branches node/compose-check (28d5a315f) and node/rcs19-freshness (5344042687) are already in flight; reuse, do not commission again. RCS20 family targets partially delivered. Native launch design #1550 and overlay design #1553 are merged design decisions, not yet runtime qualification.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. #1558 merged, #1546 patched-input freshness merged, #1557 RCS25 discrete harness, #1554 rad0065 promoted. Latest release evidence #1559 covers the two Gate E layouts on 2020.11.5, before rad0065 recipe selection. Lexical #1555 footprint intersection 37/40 packages (31 production), 45 paths: not anchor failures. Candidate has native gfx1100 IQR/int8, per-card budget, queue reserve, host pinning and loader changes. Use adoption matrix, refresh candidate when executing.

Concurrent #1563 RCS15 inventory and advisory triage reused; composition remains unproven. BU03 #1561 owns CI speed measurements; MEN24 owns actual bump phase timing only, reuse CI data, no duplicate telemetry. #1560 names release 2020.12.0; no Gate E proof verified for that release here.

Owner scope correction 2026-11-25: docs/design/radiance-native-yaml-launch.md section Scope and acceptance rule governs. Preserve behavior; smallest native launch and YAML policy change, no generalized machinery. Reuse existing tests/evidence, one shared dense/helper installed acceptance plus materially different affected paths only; cheap parser checks across approved YAMLs. Independent outstanding work stays with its owner, not a blanket S18 prerequisite. Existing bump phases, RCS20 family targets and RCS25 discrete tests stay; no artificial extra bumps or historical replays merely to close this plan. Measure improvements during actual requested bumps. Reuse existing Gate E and affected-path tests, no blanket new hardware matrix.

2026-11-25 lean build sweep at 79c61a758. One existing CMake/Ninja build graph, one canonical matched release lane, existing tests and release records. No new cache, build system, manifest, hash scheme or generic framework. Reuse actual evidence; only test changed behavior and dependency closure. Build sequence: exact selected source/patch composition once -> actual affected family/unit compile from existing graph -> RCS25 only for changed kernel rows -> canonical integrated lane when needed -> existing installed qualification, reuse proof. BU03 owns cache trust and duplicate CI work, RCS19 freshness and composed-input check, RCS20 merged family targets. Do not force extra historical transition replay or two artificial bumps to finish this work; collect evidence in actual requested bumps. Native YAML launch migration is ordinary component build work, not a new lane or build system.

2026-11-25 current-pin implementation decision: docs/design/radiance-native-yaml-launch.md §4.3 controls. Finish S18 on approved Radiance b6b8751 (2.2.0); do not wait for a bump, cherry-pick newer Fleet or extend scope. Native single-model Fleet is already present; reuse actual source behavior. S18 can land on 2.2.0 independently of next upstream release. At actual next approved bump compare 5e0cd1e2 per-card budget, 4865bd6b/4fbce0c9 multi-model queue behavior, 0158f244 ownership and 5aa5fa0f/80e094b3 against remaining narrow local hooks; adopt native behavior and retire only redundant qualified enhancement. Do not reimplement or backport these now, do not make bump a migration gate. Single-model skip-broker/skip-queue behavior verified at pin. Retain exact candidate validation and affected path testing, not historical matrix.

## Change Log

- 2026-10-21T00:30:00+00:00 (created-by): Created by agent
- 2026-11-04T03:47:59.755603+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:45:51.868013+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4603
- 2026-11-25T02:03:09.382626+00:00 (updated-by): Updated: section:title, section:notes, section:detailed_solution, section:acceptance_criteria
- RV4626
- 2026-11-25T02:37:17.678402+00:00 (updated-by): Updated: section:notes, section:detailed_solution, section:acceptance_criteria
- 2026-11-25T02:53:58.730361+00:00 (updated-by): Updated: section:detailed_solution
- 2026-11-25T02:55:02.743727+00:00 (updated-by): Updated: section:notes
- 2026-11-25T04:30:33.417688+00:00 (updated-by): Updated: section:notes
- 2026-11-25T05:21:26.606234+00:00 (updated-by): Updated: section:notes
- 2026-11-25T05:54:06.101509+00:00 (updated-by): Updated: section:notes
