---
owner-plan: RUM06
owner-group: patching-radiance-upstream
status: implementation-annex
priority: P0
work: L
skill: advanced
type: implementation-evidence
---

# RUM06 — Eliminate duplicate remote-MoE ownership and simplify Radiance patch integration

**Ownership:** Detailed implementation annex to existing RUM06, NOT a second plan or a Planning MCP state change. RUM06 remains in progress; link this annex through Planning MCP when available.

**Planning authority:** RUM06 owns remote-MoE structural cutover and lifecycle hardware acceptance. RUM02 owns shared patch consolidation and footprint tooling; RUM09 owns broad release qualification. Preserve existing RUM06 acceptance; assign these slices to one implementation owner.

**Source design and review:**
- [Remote-generation design](ENGINE-GENERATION-DESIGN.md)
- [Adversarial design review](DESIGN-REVIEW.md)
- [Detailed implementation handoff](IMPLEMENTATION-PLAN-DRAFT.md)
- [Measured deletion and reuse gates](SIMPLIFICATION-GATES.md)
- [RUM06 audit](README.md)
- [RUM02 patch footprint](../radiance-patch-footprint/footprint.md)

## Description

Replace the current BigCherry remote-MoE process-global callback and transport lifecycle with **one Engine-owned binding**, using native Radiance owner/TP group/plugin handle and kernel-instance finalisation. Remove, rather than wrap, the redundant global registry, per-call policy reads, layer-map replay, DSO static registration and duplicate close/failure logic. Reuse the existing BigCherry admission/epoch/collective mechanisms. Reduce emitted production LOC and the maintained upstream patch surface, independently measured; code relocation alone does not count as reduction.

This is a **refactor with guaranteed maintained behavior**, not a second engine, helper controller or MoE algorithm. No performance speedup is claimed without a separate measurement. The refactor is justified only when it leaves fewer ownership answers, fewer paths and less production code to maintain.

## Steps

### 0. Baseline, inventory and red tests — no production code edits

- Pin current `origin/main`, Radiance 2.2 source SHA, ABI, installed runtime identity and active patch composition before creating anchors. Record current `RUM06`/PR #1272 and RUM02's latest footprint so subsequent concurrent changes are not accidentally undone.
- Trace every production consumer of `bc_moe_remote_*`, `r3_moe_*`, `rad_bc_shared_slot`, `rad_bc_rendezvous_owner`, `Engine::serve` and mailbox kernel init/fini. Produce a **source-to-owner map** and `delete/retain/reuse` inventory; differentiate host callbacks, on-device ticket/owner wires, and typed policy.
- Run the existing `bigcherry --engine radiance patch-footprint` workflow and check into RUM06 evidence: emitted C++ LOC, authored files, non-create/upstream anchors, touched upstream files, mutable globals/lifecycle owners and exported symbols. Use RUM02's commands and existing report, not a new script/analyst framework.
- Write **failing first** card-free fixtures for: static plugin enumeration without open helper; A open/close/B reopen in same process; concurrent B helper-on refusal without altering A; helper-off C still starts; same rank different group/owner rejection; stale map and instance rejection; clean bound-ahead stop; fault wake; failed partial-init; helper session/identity mismatch.
- Gate: tests genuinely demonstrate old failure, installed ABI and ownership seams are verified. Do not proceed on assumed ABI43 line anchors. Record all expected deletions.

### 1. Consolidate policy and callback authority — remove old path in same slice

- Replace `rad0002`'s `BcMoeRemotePlugin` stateless callback dispatch with **one** size-checked context-aware internal table. Engine owns the move-only binding; plugin owns opaque transport implementation. Migrate **all** internal callers and tests together, removing obsolete exports, not leaving shims.
- Capture already admitted typed policy **once per Engine** (handoff, max rows, skip-owned, shadow/mask-only diagnostics, helper/session identity). Use a **single existing BigCherry policy predicate** for placement, MoE declare, HyperConn fusion and runtime. No repeated `bc_serving_remote_settings` reads, separate parser or global fallback.
- Move `g_plugin/g_mu/g_layer_maps/g_burst_active` into that binding; delete registration/map-replay code if the binding precedes declaration. Do not replay another model's map or hold the map lock while calling a potentially blocking plugin function.
- Bind immediately after `Engine::load_plugins` claims native `slot_base_`, using the exact `Registry::plugin("libr3moe")->handle` to resolve the one DSO factory. No second `dlopen`/loader or `RTLD_DEFAULT` registration. Reject a second helper-on engine before any helper mutation or GPU allocation; allow helper-off engines. Do not enable upstream `rad_arch_multi_instance` as a workaround.
- Gate: source-level scan proves old mutable callback registry and duplicate policy readers are gone; policy/skip-owned/HyperConn parity and repeated-generation tests pass. Record emitted LOC and added/removed attachment sites.

### 2. Reduce DSO transport and GPU mailbox lifetimes

- Convert `moe_remote_hook.cpp::HookState g` into one per-binding opaque `HookState`, with the **existing** Ledger, RankCompletion, HelperWatch, EngineRankCoordinator, rank buffers, GPU events, mmap and background loops. Delete `HookLifetime` and all constructor/destructor registration, not just rename them.
- Maintain immutable kernel row/schema enumeration at `rad_kernel_count` time: upstream `Registry::commit_` enumerates before the Engine can claim a TP group. The existing `moe_mailbox_plugin.cpp::live_count` must be deleted; helper-on eligibility happens in graph declare and kernel init, not loader enumeration.
- Use upstream `Registry::init_instance`'s supplied `tp_group` to borrow the Engine binding once per mailbox instance. Validate owner/group/world/rank; hold the device and generation borrow for the lifetime of `RadArgs::instance`. Reuse the already present `RadKernelInfo::fini` finaliser to release it, not a new lease framework.
- Investigate `rad_bc_shared_slot`/native owner scope first for non-owning group lookups. Only if they are unsuitable, add one **small non-owning group→binding lookup**, used outside the token/route/combine hot path. It must not hold state, policy, callbacks or resources and must be erased before group reuse.
- Change direct `r3_moe_mailbox_device`, owner `upload/ready`, placement `close/wait/reopen` and all epoch coordinator functions to use that exact same bound instance. Remove unbound or rank-only old variants. Keep the existing 16-byte StepTicket and separate native Engine owner, group, helper session and placement epoch.
- Gate: DSO load/unload has no helper startup/shutdown effects; no stale pointer or second model takeover; named failpoints leak no fd, mmap, pinned host/device allocations or events; no lookup/allocation added to normal hot path.

### 3. Unify release ownership; split clean finish from fault wake

- Clean finish: call existing ledger `finish_admission`; cancel only unused unbound tail; finish any bound current/ahead parents (rank seal, terminal events and host_done exactly once). Wait existing helper + rank proof, join worker/snapshot/placement threads and `Engine::drain_streams` including transfers and owner upload events. Then use **native Registry instance `fini`**, detach callback/group index, release transport. No successful close may call `ledger.interrupt` early or abort the rank barrier before clean joins.
- Fault finish: poison existing Ledger and TP rendezvous, wake blocked admissions/peers, abort appropriate barriers, unwind/join ranks and GPU readers, then use **the same release tail**. Preserve the failed helper/session and original error rather than claiming success or falling back to local expert compute.
- Partial start: use exactly that same release tail for every acquired resource, but do not free published peer pointers until the other rank participants have unwound. Reuse rad0017 `free_rank_buffers`, deferred cleanup and `bc_shared_slot_retire_owner`. Preserve all AQL acquire/wake/doorbell/chain-fence semantics.
- Gate: injected failures before/after published pointer, all alloc sites and peer timeout produce zero leaks, zero double-frees, no UAF, correct device-scoped frees and no hangs. A clean Engine can be followed by another same-process generation without stale maps or resource reuse.

### 4. Simplify managed source and patch interface using RUM02

- First **delete obsolete code**, then consolidate surviving injected `rad0002` C++ currently held as large `patch.py` string literals. Adopt RUM02's existing `src/`-owned SHA-256-pinned source pattern from rad0016/rad0017, or an existing engine component compilation unit where reuse is better. Do not create a new source-reading framework.
- Keep only the narrow typed engine/arch hook attachment sites; validate actual anchor counts, affected upstream files, source hashing and composed exact ABI44 build. Do **not** edit historical/archive packages; coordinate extraction within RUM02 rather than reopening already assigned consolidation work.
- Measure semantic emitted C++ deletion separately from moving C++ text out of Python; report both authoring footprint and upstream-anchor deltas. No performance rewrite, kernel math/fences, new config representation, alternate helper operation or duplicated generic.
- Gate: explicit before/after LOC and patch-surface report; **emitted production LOC deleted > added**, fewer mutable singletons and lifecycle owners, no new duplicate paths. If a necessary safety invariant makes net LOC positive, stop with a measured variance and exception request; don't cheat by removing tests, error messages or invariants.

### 5. Focused verification, integration and handoff

- Execute existing host fixtures (not just grep/string tests), exact ordered patch application, idempotence, `patch-rebase-check --all`, `patch-footprint --check`, libr3 rewrite `--check`, CMake/C++ ABI and actual `nice -n 19` builds **one build at a time**. Candidate/source pin and build identity must be logged. Retire existing tests asserting that `g_plugin` and `HookState g` remain; replace with behavior checks.
- Run the four named mixed-TP hardware proofs already required by RUM06 under **short GPU leases**: per-rank gfx code objects, reversed visible ordinals, same/different TP-group rendezvous and timeout, and zero-width/ragged span ownership. Use actual built candidate and installed 1170.37.0 control where applicable. Record lease receipts and numerical behavior; do not treat older release logs as proof of the refactor.
- Run one representative helper-on normal start/steps/clean stop/restart and helper-off control on the candidate. RUM09 retains all-config release/KLD qualification; do not duplicate that campaign here.
- Review actual production diff against every critical `DESIGN-REVIEW.md` finding and deletion ledger; commit source/test/evidence in coherent slices. Coordinate plan handoff with RUM06 so only RUM06 implements the structural cutover and RUM06 validates resulting behavior.
- Gate: all host/hardware gates recorded with commit, source pin, exact commands, exit codes, logs and comparison; linked review findings resolved. Keep RUM06 in progress until these pass.

## Detailed Solution & Technical Design

**Target ownership, not a new framework:**

```text
Radiance Engine (owner_, slot_base_/tp_group, typed admitted settings)
  ├── existing BcMoeRemoteAdmission: existing Ledger parent/ticket flow
  └── ONE RemoteBinding: immutable policy, callback table, per-gen maps/burst
       └── ONE opaque libr3moe HookState (existing rank/epoch/watch/GPU state)
            ↑ borrowed by kernel instance through native tp_group
              ↓ released by existing RadKernelInfo::fini
Radiance Registry: exact RTLD_LOCAL DSO handle, static kernel rows
rad0017: existing shared group rendezvous, deferred peer ownership/retirement
```

**Never add:** a second `EngineExecutionState` service, helper scheduler, parent-ticket registry, placement epoch, callback bus, generic `KernelLeaseManager`, mutable global policy, special `rtld` loader, compatibility C ABI, another collective rendezvous, or a fallback that executes an expert under ambiguous ownership.

**Internal ABI:** a single opaque generation plus a size-checked table whose functions take a transport instance. Do not replace a direct callback ABI with two layers of lookup. Frozen policy must be passed as a value/typed owned snapshot to every decision point, including `bc_moe_remote_block.h` and HyperConn; merely moving `g_plugin` does not solve global policy. Upstream `Geom` is not itself a remote-state owner; pass only the small needed policy view through BigCherry-controlled extension code. No Engine pointer in GPU tickets.

**Sequencing invariants:** plugin schema/kernel discovery occurs before native group claim; activation follows group claim and precedes graph declaration; rank/host/GPU proof precedes instance finalisers; finalisers precede transport free; shared group retirement follows peer unwind; exclusivity claim releases last. Fault wake and successful drain are different entry paths to a single cleanup tail. Close/reopen of a **placement** epoch is not Engine terminal close.

## Code Samples & Guidance

The existing RUM02-owned source-hashing pattern should be reused, not rewritten. Existing test suites and commands to run as applicable:

```sh
python3 -m pytest -q tools/tests/patch/test_rum06_registry_lifetime.py
python3 -m pytest -q tools/tests/patch/test_vtp14_n3_shared_rendezvous.py
python3 -m pytest -q tools/tests/patch/test_vtp14_n2_tp_ownership.py
bigcherry --engine radiance patch-footprint --check
bigcherry --engine radiance patch-rebase-check --all
# Build exact candidate, with one compile at a time:
nice -n 19 cmake --build <prepared-build-tree> --target <focused-components> -j1
ctest --test-dir <prepared-build-tree> --output-on-failure -R '<focused-lifetime-fixtures>'
# On explicitly granted GPU lease only, use actual installed test entry points:
agent-lease acquire gpu --ttl 15m --est 10m --wait
```

Commands shown are **planned gates**, not execution claims. Confirm CLI syntax against current `bigcherry --help` at test time, and discover actual current build target names from CMake. All test logs under owning RUM06 evidence. Prior 3/3 `test_rum06_registry_lifetime.py` proves only the old registry contract and is **not** acceptance for the new design.

## Files

**Primary implementation, exact owners:**

- `engines/radiance/patches/rad0002_moe_remote_handoff/patch.py` and its eventual existing-pattern owned `src/` files, or existing component sources when appropriate.
- `engines/radiance/components/moe-remote/moe_remote_hook.{h,cpp}`, `moe_mailbox_plugin.cpp`, `moe_mailbox_ops.hip`, `CMakeLists.txt`.
- `engines/radiance/components/serving/bc_engine_settings.h`, `bc_serving_manifest.cpp` **only if** needed to access the existing admitted typed snapshot, with no new reader.
- `tools/tests/patch/test_rum06_registry_lifetime.py`, existing CTest fixture targets and new focused lifecycle tests in the **same existing test owner**.

**Read/verify first; modify only if a red test requires:**

- `engines/radiance/components/moe-remote/admission_ledger.h`, `admission_completion.h`, `engine_admission_gate.h`, `engine_rank_coordinator.h`, `owner_generation.h`.
- `engines/radiance/patches/rad0017_vtp14_n3_shared_rendezvous/{patch.py,src/bc_ar_rendezvous_impl.h,src/bc_ar_lifecycle_impl.h}`; downstream `rad0016`–`rad0033` TP/one-shot/fence patches.
- `docs/evidence/radiance-patch-footprint/{footprint.json,footprint.md}` via **existing** RUM02 tooling, not manual numbers.

## Validation

| Gate | Required proof | Current |
| --- | --- | --- |
| C1 single policy/ownership | No global `g_plugin`, `HookState g`, DSO constructor, duplicate getters; exact bound group/owner/session | Not run |
| C2 loader and direct ABI | RTLD_LOCAL factory; stable kernel rows; strict context; no stale rank/owner/device call | Not run |
| C3 sequential/concurrent models | A→stop→B same process, B concurrent refused without altering A, helper-off C safe | Not run |
| C4 admission/stop/fault | Bound ahead completion, only unused tail cancel, helper-drain vs fault wake, no false success | Not run |
| C5 partial-init cleanup | Every injected allocation, map, watch, post-publication failure; no leak/UAF/double free | Not run |
| C6 TP collective continuity | Cross-DSO group, same/different-group isolation, zero-width/ragged shards, unchanged fences | Not run |
| C7 simplification | Net negative emitted C++ LOC; removed duplicated control paths; maintained patch anchors measured | Not run |
| C8 builds/replay | ABI44, actual code compilation, whole active patch replay, RUM02 footprint check | Not run |
| H1–H4 mixed-TP | Fresh code-object, ordinal, rendezvous, span hardware receipts and canary | Not run |

Evidence for the refactor cannot be inherited from v1.17, old ABI43 fixtures or a former TP3 launch. Report pass/skip/fail distinctly.

## Effort & Risk

**P0, L.** Highest risk: kernel enumeration before Engine binding, lifetime of captured GPU instance pointers through close, multi-owner races, and stale group reset after a failed peer. The refactor must not expand patch surface or bundle a new generic system with its own lifecycle. A design is rejected if it eliminates one global by adding a second registry, duplicates policy or introduces two shutdown APIs doing the same cleanup. Treat any net-positive production LOC as a mandatory redesign/exception checkpoint.

## Standards

Follow `docs/reference/agents/WORKFLOW.md`, `docs/reference/tooling/TOOLING.md`, `docs/reference/patches/PATCH_SYSTEM.md`, existing RUM02 footprint/source-hashing standards and the ordinary planned PR/review/evidence process. Implement on an isolated task worktree from current main when authorized by RUM06; do not edit shared worktree, archive patches, generated plan state, ledger or release snapshots. No legacy, shims, hand-picked constants, extra model parsers, host names/paths or attribution lines. Short GPU leases; one nice-19 build at a time; only kill owned PIDs. Keep source, ABI and provenance evidence reproducible.

## Acceptance Criteria

1. **Less production code and less ambiguity:** measured emitted production C++ changed by this work has **more lines removed than added**; source moves alone do not count. Show before/after patch anchors/upstream targets, mutable-global owners, callback paths and exported symbols. No duplicate compatibility/fallback path. An exception requires an explicit owner decision with the measured invariant and alternatives, never silent waiver.
2. **One owner across every entry point:** all remote callbacks, policy, layer maps, mailbox GPU pointers, owner upload/ready and epoch transitions bind to the same Engine owner, TP group and helper/session generation. No DSO constructor registration or mutable process-wide execution state. An unavoidable single non-owning exclusivity guard is allowed.
3. **Reuse rather than replace:** native Registry/DSO handle, `RadKernelInfo::init/fini`, native Engine ownership, existing Ledger, epoch coordinator and rad0017 cross-DSO shared-slot/release are used as the **only** state authorities. No second scheduler, owner registry, epoch service, collective or generalized plugin loader.
4. **Correct lifecycle:** clean stop closes admission and drains bound rank/host/GPU/helper work before detach/free; faults wake all blocked peers, preserve the named cause and unwind safely; partial-init failures do not leak or free peer-visible pointers; same-process repeated generations and concurrent second helper-on refusal are proven.
5. **Production and hardware evidence:** focused red/green tests, exact current managed patch replay/idempotence, actual ABI build, four named leased mixed-TP proofs and one helper-on+off canary meet RUM06's retained safety and quality constraints. No GPU acceptance by proxy.
6. **Planning/PR handoff:** authoritative RUM06 updated through Planning MCP; RUM02 extraction ownership preserved; linked reviews closed; unreleased Ledger MCP event and verified PR checks and merge recorded before RUM06 completion.

## Notes

**Why this annex:** RUM06 accumulated partial fixes; this focused implementation breakdown makes deletion, reuse, and maintainability mandatory while retaining one authoritative plan. RUM02 retains generic patch-source consolidation.

**Planning action pending:** Through Planning MCP, link this annex and acceptance gates to existing RUM06 and reconcile RV4303. Do not hand-edit Planning-managed state or create another RUM item.
