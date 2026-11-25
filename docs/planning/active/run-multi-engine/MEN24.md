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

Order A–H (one programme checklist, no new plan item): A exact candidate and current-pin self-triage, RCS23 origin/row/dependency and native interface impact; B MEN16/PA49 declare/pull/audit and pristine coupling; C RCS19 exact recipe patch materialization then authoritative rewrite/ownership checks; D RCS19/20/21 affected host/device compile and link/export, actual transitive closure or conservative full set; E MEN16 canonical lane build/reuse plus RSC01/02/05/06 and RSP01 matched native parser, all YAMLs and model/row roster; F MEN15 native CPU smoke and separate card-free owner tests; G RCS25 discrete affected-unit tests (RCS23), MEN17 exact candidate release acceptance, RSC11 lifecycle and PWR10 qualified power restoration; H reaudit, bound evidence, explicit upstream adoption disposition and RSP01 deliberate release. Existing pin-bump phases and adapter interface are sole orchestrator; exact phase names may be adapted to existing state machine. RCS15 provider decision gates large gfx11 overlay changes, not early gate work. RSC11 S18 native launch and PWR08 converge in one cutover, no interim manifest power producer. RCS19 gate integration, RCS25 harness, MEN15/16/17 binding and RSC11 can run in parallel; no separate programme. Nightly read-only exact-ref triage with changed dependency/row/native boundary report, no repin or qualification.

## Code Samples & Guidance



## Files



## Validation

The next radiance bump records per-phase times; target under 15 minutes end to end when no rewrites drift; one engine build per bump; a radiance bump runs no llama.cpp tests; the nightly drift job reports on upstream main.

## Effort & Risk



## Standards



## Acceptance Criteria

Retain all earlier speed and correctness gates; prove exact staged order, no required skipped/unverified proof accepted, phase result identities and invalidation, same selected recipe and candidate build. Record per-phase timing, compiler hits, affected/full-fallback unit counts, duplicate build avoidance and human reconciliation time; compare at least two real candidate bumps through RCS23, including resume/rollback and qualified release receipt. Earlier under-15-minute target is for clean software path, not GPU queue/qualification guarantee. Report actual timings, no claim from card-free alone. Cross-engine unchanged; no new driver.

## Notes

Measured 2026-10-20 from the .ninja_log files of lane build 4babe498 and a local run of tools/tests/patch. Related: MEN10-MEN19 (managed engine workflow), RAD15 (engine rows instead of rewrites).



BACKLOG TRIAGE 2026-11-04 absorbed items (superseded into this one; their notes keep the full history): MEN19: pin-bump has the drift phase (NATIVE_REWRITE_DRIFT) and caught the v1.16 rewrite break; remaining pin-release confirmation work is MEN24.



CORRECTION 2026-11-04: the earlier note that MEN19 were absorbed into this item is withdrawn. An independent revalidation found each either still has open scope of its own (reopened) or was delivered outright. This item's scope is unchanged.

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. Existing 1.9.0 timing and compiler-cache goals retained as historical evidence; not a measured current result. RCS19 branches node/compose-check (28d5a315f) and node/rcs19-freshness (5344042687) are already in flight; reuse, do not commission again. RCS20 family targets partially delivered. Native launch design #1550 and overlay design #1553 are merged design decisions, not yet runtime qualification.

## Change Log

- 2026-10-21T00:30:00+00:00 (created-by): Created by agent
- 2026-11-04T03:47:59.755603+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:45:51.868013+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4603
- 2026-11-25T02:03:09.382626+00:00 (updated-by): Updated: section:title, section:notes, section:detailed_solution, section:acceptance_criteria
