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

### Active enhancement parity (2026-11-27)

Follow [enhancement parity code/test guide](../../../design/radiance-enhancement-parity.md). Maintain every active capability, including diagnostic/optional ones outside current YAML. Derive old/current inventory from actual patch metadata and recipe selectors, rewrite blocks, origins and components; record grouped item-level decisions and evidence in existing bump README, not new ledger format. Reject incomplete native replacements by source counterexample before testing; use minimal old-contract / same-pin native vs retained controls only where credible. No removal/demotion without explicit proven successor or owner-approved capability change. Archive code/evidence intact. Final release gate is both enhancement parity and current-model compatibility, not either alone.

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

2026-11-27 sync review RV4701: merged main 437799b4e without conflict (release manifest fix #1724). Production patch rebase 33/34 clean, one remaining rad0002 MoE handoff. ABI45 and TP/collective/pipeline/KV host checkpoints recorded in ../../../evidence/2026-11-27-radiance-230-bump/IMPLEMENTATION.md. Next: rad0002, affected native rewrites and owned providers; retire only unsupported encoding-exclusive code with source/evidence archived (RETIREMENT.md); retain plain MXFP4 drafter and mixed INT8/IQR paths. Then exact canonical build reuse and final candidate-bound roster/quality/performance gates. No entire plan is complete; do not count source composition as runtime qualification.

2026-11-27 first executable gates: base candidate e6598253c linked base/BC overlays/libr2. Native CPU YAML smoke and converted pinned IQ8R fixture pass real deterministic completions; gfx1201 base IQR/int8/IQ8R/logits 205 distinct checks, zero failures; GPU tiny fixture refused h64 attention (CPU-only, not production geometry). libr3 DS41 gfx12 prefetch-only codegen port; full link and exact candidate ABI, overlay/native GPU parity and production model graph/lifecycle remain. Evidence FIRST-RUNTIME-TESTS.md.

Owner request 2026-11-27: include MRC08 portable benchmark records in current Radiance 2.3 bump. MEN17 owns matched independent baseline/candidate comparison and promotion gate; MRC08 owns capture/export. Integrate reviewed MRC08 through normal PR flow before comparison. Retain exact old/candidate build identities, collect necessary matched cohorts, bind both run records and comparison verdict to formal bump receipts. Do not rerun passed compile/smoke gates merely to populate exports. See MEN17 RV4702.

Baseline first: 71 ABI45 DSOs, 93 offline tests, actual four-container/18-YAML preflight and smoke on prior canonical build; RSC19 matched owner build and VTP59 screens subsequently verified. Latest origin/main marks MEN29 completed. Integrated RSC19, VTP59/60 and MRC08 commits into bump; 117 tests/24 subtests passed. Actual release acceptance still uses self-recorded helper KLD and ambient binary; no qualifying release evidence. Shared KV guide L requires new versioned plan/receipt, retained host layout, physical allocation and scheduler boundaries; not a cheap pin reconciliation. Do not conflate native format tool or phase trace experimental patches with production. Baseline-first controlling scope: docs/evidence/2026-11-27-radiance-230-bump/BASELINE-SCOPE.md. Preserve existing phase acceptance. Shared KV RSC16-18 new mode remains post-baseline; only legacy rank/page/refit/accounting fixes needed to preserve current approved configurations belong in this bump.

Owner approved focused historical recovery pass (RV4755): target TP3 exact seed1 512/192 aggregate >=353.615 t/s (within2% of legacy360.8315, provisional; historical 8-stream decode noise0.26ms is not aggregate noise). Preserve TP2 gains, matched TTFT/gap, quality and exact wire; prior TP2 W is lossy. Match older RUM06 4K/256 as additional bridge. First bounded configuration screen pipeline floors128/256/512 on same enhanced build, 75ms target, 2048 batch, FP8, no prefix, stock observe. Existing all-reduce piece sweep rejected one-piece; don't repeat. Native architecture already has shared quantized activations, norm/add/quant and GDN/gated fusion; verify active row eligibility before new owned enhancements. No new shared KV/streamed/arena or scheduler architecture in bump; only promote confirmed win with repeat controls and independent quality. Correct previous initial mixed cap: 512 on real TP3 2048 batch; 128 was smaller host fixture.


Owner decision 2026-11-28 (supersedes the RV4755 provisional bump target): the 2.3.0 performance gate is no regression against the installed 2.2 release bc-radiance-2020.24.1, not recovery of the historical 2020.8.5 figure. Same-session evidence already in this bump: TP3 IQ8R 8x512/192 candidate 342.4-344.5 t/s vs installed 2.2 351.3, decode ms/step 36.75 vs 37.36 (candidate faster per step), draft acceptance 0.70 vs 0.70; 4K/256 107.1 vs 106.6. Most of the deficit against 2020.8.5 (360.8) predates the bump (2.2 is already -2.6%). Remaining bump gate: one ABBA candidate vs 2020.24.1 under one lease, several seeds, decode ms/step and TTFT/prefill reported separately (not aggregate t/s alone); a confirmed prefill/admission regression is fixed narrowly in the bump, otherwise pass. No further historical-recovery or tuning screens in the bump. Post-bump follow-ons: VTP61 (TP3 short-prompt history bisect 2020.8.5 -> 2020.24.1), VTP59 historical recovery, RCS23 IQ8R prefill row comparison (upstream r3d_gemm_i8a8 vs rad0061) on the merged 2.3 build, then RCS21/RCS23 ownership decisions.

Final corrected canonical f3b75ae7e258f3a0c75870e20060a40f versus installed 2020.24.1: ABBA 24 trials, 3 matched seeds, 8 streams, exact effective settings and token counts, zero preemptions, clean stop. Evidence installed-regression-abba.json. Pooled short 343.4005 to 350.47 t/s, long 107.3455 to 106.904 t/s; paired seed relative medians +0.786%/-1.424%; pure decode ms/step 36.444 to 36.0885 and 36.74305 to 36.6032. Long p95 gap pooled flat but paired seed +9.956%, old-launch variability; no conclusive latency recovery. No material new throughput/decode regression demonstrated in this cohort. No change to tuning defaults. Diagnostic, not formal release admission. Historical recovery remains VTP61 post-bump per owner scope. Exact canonical 71 ABI45 plugins and 93 component tests pass.

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
- RV4701
- 2026-11-27T06:12:42.519052+00:00 (updated-by): Updated: section:notes
- 2026-11-27T06:14:45.981191+00:00 (updated-by): Updated: section:notes
- 2026-11-27T06:15:56.224380+00:00 (updated-by): Updated: section:notes
- 2026-11-27T08:01:27.930195+00:00 (updated-by): Updated: section:notes
- 2026-11-27T09:21:15.704257+00:00 (updated-by): Updated: section:notes
- RV4754
- 2026-11-27T15:10:03.429238+00:00 (updated-by): Updated: section:notes
- 2026-11-27T22:07:24.087969+00:00 (updated-by): Updated: section:notes
- 2026-11-27T23:01:45.598946+00:00 (updated-by): Updated: section:notes
- 2026-11-27T23:13:03.679997+00:00 (updated-by): Updated: section:notes
