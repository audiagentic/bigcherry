---
id: RUM14
order: 14
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-22T08:21:43.996915+00:00'
breadth: ''
skill: advanced
created-by: gpt-6
priority: P1
work: L
---

# Use one production native container-weight materializer for declared-graph tests

## Description

Radiance 2.2 / ABI44 already owns ONE native loader: Engine::load_weights in core/engine_bringup.cpp obtains resident destinations and calls plan_container_weight, RelayoutPool and sweep_container; refill_weights reuses the same path. The LTX02 model-backed C06 test proves all 10 declaration weights per rank, 2 nonidentity IQ8R scale relayouts, and 24 source-backed native hook calls but has not loaded resident weights into RadCtx. This RUM14 owns only a generic reusable production declaration-to-resident-weight seam. LTX02 exclusively owns LTX Attn/Act/Lin issue, numerical oracle, video/audio quality. RUM13 already owns AQL exact collective qualification; RUM06 owns group lifecycle; RUM05 owns admission. No duplicate loader, scheduler, registry, config reader, transport, or reopening completed plans.

## Steps

1. READ/RED GATE: inspect frozen ABI44 core/engine.h Engine::start, core/engine_bringup.cpp Engine::load_weights/refill_weights, plan_container_weight, RelayoutPool::run and sweep_container, core/format/share.h and runtime/ctx.h. Record exact ownership, LOC, native ABI, source model and all existing APIs. First determine whether one existing public native path can load prebuilt Builder/Program weights. Write a FAILING test of missing C06 resident weights; do not equate the existing 24 relayout calls with residency.
2. PREFER REUSE. If Engine can accept a preadmitted test Program without bypassing arch restrictions, reuse it. Otherwise extract the smallest generic native core-owned batch loading seam that BOTH Engine load/refill and tests call, moving and deleting old source-planning/sweep/staging code rather than copying it. Inputs: native Program/WeightInfo, RadFile, rank/world/device and placement-approved destinations. Reuse the existing weight_share, plane rectangles, RelayoutPool, PinRing, file-ordered sweep and kernel-selected layout hook. No new public ABI or patch unless essential and reviewed; never add work to per-step issue.
3. PRODUCTION SAFETY. Preserve mapping/SSD/pinned/VRAM/mover ownership, checkpoint fallback, refills, tiny scales, output size and allocation, source-file extent checks, send/wait ordering, failure-atomic rollback, and canceled load. Pre-publication error must not leave stale Program pointers. Add missing negative cases (wrong rank, missing source, truncated plane, stale/missing relayout, early stop/cancel and failed upload) BEFORE changing the shared code.
4. MODEL HOST CONTRACT. In the existing C06 Builder fixture load every model weight for all 3 ranks, video and audio block zero and both 8/8/16 and 12/12/8 legal head maps, using the one shared generic load path. Verify 10 real source-backed stored weights per rank and the 2 nonidentity relayout scales, with mutation-sensitive comparisons. Delete test-only duplicate plane gathering; source and stored bytes must agree with the unmodified production loader, not a hand-rolled oracle.
5. C06 ISSUE CONSUMER. After generic loader acceptance, LTX02 extends its EXISTING C05 RadCtx and C06 Attn tests (no independent graph or launcher) to issue x.Act, fused IQ8R Lin QKV/gate, full Q/K normalization, RoPE, attention, output collective and exactly one bias. Capture TP1 video/audio intermediates, compare within an explicitly approved tolerance. Use RUM13 AQL ranks and RUM06 group lifetime; reversed card order, 1/16/64 rows, missing rank, bounded timeout and clean restart.
6. BEFORE MERGE: inventory production/test added and deleted LOC, eliminated paths and patch count, no per-step changes. Verify exact 2.2 patch apply/idempotence, host component tests, production memory-tier/refill/SSD tests and real-model native C06 regression. If engine loader code changed, require one installed same-SDK mixed-GPU serving canary under a short exclusive lease, with source/weight identity and startup/transfer timing; do not sweep unchanged configs. Keep production LTX world_size>1 refusal until LTX04.

## Detailed Solution & Technical Design

The core issue is NOT a missing collective: C05 native RadCtx already passed 2,088,960 exact BF16 comparisons with source gain on three cards and C06 native AQL initialized 108 instances for each reversed card order. The missing ownership boundary is source-model weight residency. In the production core, core/engine_bringup.cpp:3863 plan_container_weight constructs LoadRun and StagedWeight; 3200 RelayoutPool::run invokes weight_planes and the kernel layout (or identity and widen/narrow); 3369 sweep_container reads in file order; 3923 Engine::load_weights owns the placement/mover walk; 4259 Engine::refill_weights reuses the same planner and sweep. Core Engine::start constructs the whole model and cannot take a prebuilt test Program through its public start interface. Prove whether there is another smaller existing entry point before extracting.

Conceptual generic ContainerLoadBatch (name is illustrative, not an API commitment): accept already-validated native Program weights and rank destination pointers, use exactly the production planner and pool, publish only after complete success. Engine remains the sole placement, host memory tier, device code and checkpoint identity authority. Host native tests are bounded consumers of that shared loader, NOT a second file reader or IQ8R layout algorithm. If moving the whole pool would add maintenance paths or duplicate 500+ lines, stop and redesign. No wrapper that simply calls the same callback on test-only gathered plane bytes qualifies. Preserve native ABI and ownership, no per-step lock/allocation. Reuse the frozen current-source patch conventions.

## Code Samples & Guidance

RadFile + frozen Program and WeightInfo + validated rank-share + declared destination -> native production plan_container_weight / RelayoutPool / sweep_container -> identity or native IQ8R relayout + upload -> RadCtx. LTX02 owns only next Act/Lin/attn_core issue and TP1 reference. Explicitly forbidden: independent LTX plane reader/rectangle gather, raw copy of nonidentity rad.planes, second transport or per-step loader.

## Files

Frozen Radiance core: core/engine.h, core/engine_bringup.cpp, core/format/radfile.h, core/format/share.h, core/runtime/ctx.h, core/build/rad_build.h. Source-managed BigCherry patches: engines/radiance/patches/, recipes.toml, exact patch-unit tests. Existing LTX C06/C05 consumer: tools/lab/radiance/ltx-tp3/native-c01-gate/ltx_c02_lin_declare_test.cpp, ltx_c05_issued_test.cpp, CMakeLists.txt; existing diagnostic rad0060_ltx2_tp3_linear_declare (unselected, source immutable unless separately justified). Evidence: docs/evidence/2026-11-22-ltx25-ltx02-c06-weight-contract/README.md, docs/evidence/2026-11-22-ltx25-ltx02-issued-aql-rms/README.md, docs/evidence/2026-11-22-rum14-native-container-weight-seam/README.md. Generic RUM14 only; LTX02 is model-specific.

## Validation

Host red/green mutation-sensitive: actual real IQ8R plane byte extents and 24 relayouts/12 rank-modality-split cases, stale layout/missing source/wrong rank/short plane/cancel, no pointer publication on error. Existing native CTests C05 issued QK, C06 native weight-share, C06 host instance rendezvous. Source-pinned patch-rebase-check + component compile + production serving/refill/SSD/host/VRAM regression. For any production loader change a leased installed unchanged-config three-card smoke and both native AQL orders, evidence exact source/build/model hashes and abort behavior. No GPU needed for design-only step, no TP1 or generation claim until actually issued.

## Effort & Risk

P1; medium-high or large, first API/design slice bounded. Main risk: a claimed tiny public helper forks engine_bringup source reading, data layout, cancellation, tier ownership and refill. If no safe shared extraction reducing code/path count is found, STOP with source-proven evidence and change the RUM14 design, not the production ABI. Do not enable LTX TP3 or add a hotpath check.

## Standards

BigCherry docs/reference/agents/WORKFLOW.md and docs/standards/ARCHITECTURE_STANDARDS.md; source-managed patch tests, core/serving component manifest, docs/reference/tooling/TOOL_DISPOSITION.md, evidence/release conventions, no source-file local host paths or hand-picked model constants, native ABI44, exclusive GPU lease. Change Planning/Ledger only with canonical MCP. No shims or duplicate loader.

## Acceptance Criteria

There is a single, engine-owned native production weight-materialization path called by ordinary Engine load/refill and by the bounded C06 declared-Program test; source shares and kernel layouts are native. All 10 C06 weights per rank resident including the two IQ8R nonidentity scales, source extents checked, no partially published output on error, and test-only duplicate read/relayout staging removed. Production refactor demonstrably reduces competing paths and adds no per-step work; exact 2.2 component, layout, refill, source-replay and installed canary tests pass. LTX02 thereafter proves complete issued self-attention, TP1 parity and fault gates separately. TP3 serving remains refused.

## Notes

2026-11-22 source audit against main 06d5831a0 / bc-radiance 2020.1.1. Existing LTX C05 full three-card Q/K 2,088,960/2,088,960 BF16 exact is seeded activation with real gain, not TP1 captured; C06 host 24 IQ8R relayout hook calls and 4 graph groups / 108 initialized AQL instances per device order, no resident checkpoint C06 or generated video. Scope split: RUM14 generic source loader only; LTX02 (active) model-specific issue/parity; RUM13/RUM06 completed. Details in docs/evidence/2026-11-22-rum14-native-container-weight-seam/README.md.

2026-11-22 after PR #1315 (c72ed6be3): the C06 native test now genuinely ISSUES one checkpoint-weight, single-row block-0 video/audio self-attention in four legal cases, all three ranks, with both mixed AQL card orders passing (108 native instances/order; bitwise equal output across orders). The diagnostic uses bounded test-only c06_gather and direct host/GPU weight pointer/upload staging, not production Engine::load_weights, RelayoutPool or sweep_container. This is a useful LTX02 execution proof, NOT completion of RUM14. RUM14's current generic deliverable is to delete and replace that temporary test-specific weight materialization by reusing one extracted production engine loader; preserve the working issue tests and require net reduction. Avoid implementing an alternate LTX loader or reopening RUM06/RUM13. PR #1316 repairs an actual allocation unwind bug in that diagnostic. No TP1 parity or production TP3 admission is accepted by these results.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Candidate 7244cef0/1c3584a8/5aa5fa0f changes core/engine_bringup.cpp staging completion, host-only bounded reconstruction/refill and failure lifetime. Before extraction re-read approved target native loader and existing APIs, avoid backport/clone stale 2.2.0 sweep code; preserve copy completion and host/refill accounting. Existing native LTX C06 consumer and safety gates unchanged; no dependency on optional provider adoption.

## Change Log

- 2026-11-22T08:21:43.996915+00:00 (created-by): Created by gpt-6
- 2026-11-22T08:36:23.727103+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4648
- 2026-11-25T02:48:37.694195+00:00 (updated-by): Updated: section:notes
