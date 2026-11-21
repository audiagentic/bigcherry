---
proposed-item: RUM06
plan: patching-radiance-upstream
state: draft-not-planning-authority
priority: P0
work: L
skill: advanced
review-of: RV4303
---

# Refactor remote MoE into the existing Engine/TP lifecycle

**Planning MCP handoff, not a new plan item.** Preserve RUM06's ID and `in_progress` status; do not open a parallel RUMxx or replace the review RV4303. Review [design](ENGINE-GENERATION-DESIGN.md), [adversarial findings](DESIGN-REVIEW.md), [audit](README.md) and original [RUM06](../../planning/active/patching-radiance-upstream/RUM06.md). This document has the same Description / Steps / Detailed Solution & Technical Design / Code Samples & Guidance / Files / Validation / Effort & Risk / Standards / Acceptance Criteria / Notes sections as existing BigCherry plan items, but cannot be published through Planning MCP from the current tool connection.

## Description

Replace the process-global BigCherry remote-MoE lifetime with one explicitly Engine-owned binding, **while shrinking the owned production code**: remove the singleton registry, duplicate policy reads, DSO constructor, per-rank implicit device lookup, duplicate cleanup branches and obsolete map replay. Reuse Radiance's existing Engine owner/TP-group, Registry/DSO handle, kernel `init/fini`, BigCherry's Ledger/RankCompletion/EngineRankCoordinator and rad0017 shared-slot/peer-unwind implementation. Implement clean admission close/drain separate from fault wake. Prove all RUM06 acceptance clauses, including mixed-TP hardware tests.

**Change policy:** An addition is not accepted simply because it works. Each new type, API or file must name the old code it deletes or consolidates; any generic wrapper must replace at least two existing implementations. No legacy/shim/multi-path code, new independent scheduler, duplicate identity/epoch, global mutable policy, hard-coded model geometry, new standalone test infrastructure or gratuitous upstream patch. Production LOC must decrease across the migration unless an explicitly justified safety exception is approved.

## Steps

### S0 — pin facts and establish red fixtures (host-only)

1. In existing `node/rum06-complete` worktree, record HEAD, exact Radiance 2.2 source/ABI/build identity and prior audit. Confirm expected `Engine::load_plugins` / `Registry::plugin(handle)` / `rad_engine_claim` / `tp_group` / `Registry::fini_instances` contracts against the actual build source; source mirror is not itself an installed-build proof.
2. Add focused **failing** tests, without changing production, for: static kernel rows independent of live helper; same-process Engine A close / B reopen (same native slot allowed, distinct owner); concurrent helper-on B refusal with A unchanged; no-helper C; bad `struct_size`; stale rank/group/device pointer; clean stop with current + bound ahead and cancel unbound tail; fault wake; prepublication and postpublication peer failure.
3. Enumerate old functions and deletion targets: `rad0002` global `g_plugin/g_mu/g_layer_maps/g_burst_active` and all duplicated callback wrappers, `RemotePolicy remote_policy` repeated reads, pre-registration map replay; `moe_remote_hook.cpp` `HookState g`, `HookLifetime`, static registration, direct global `r3_moe_*` calls. Count physical lines, live global owners, injected upstream anchors using existing patch-footprint tools.
4. **Gate:** tests first red where intended, verified adapter seam and documented baseline metrics. If the installed ABI differs, revise design before cutting code.

### S1 — delete implicit engine registry and freeze policy

1. Evolve **one** BigCherry-owned `abi/bc_moe_remote.h` internal table into context-aware callbacks (`void* instance` + strict struct size). Remove old no-context variants and their old fake/test definitions in the same commit. No parallel protocol.
2. Add a small, move-only `RemoteBinding` member next to the existing `BcMoeRemoteAdmission` in `Engine`. Use **existing** `Plugin::handle` to resolve the DSO's open/close factory once after `load_plugins` allocates `slot_base_`. Validate one live helper, exact model/session identity, typed immutable settings; refuse missing/invalid factory before declaration/weights. Helper-off never opens helper.
3. Move `g_plugin/g_mu/g_layer_maps/g_burst_active` into that binding. Store all maps once; if the binding precedes declaration, **delete** map replay, not rewrite it. Replace repeated global callback read-lock-copy-unlock boilerplate with one generation-local dispatch (no mutex on per-step hot path).
4. Change BigCherry `bc_moe_remote_place.h`, `bc_moe_remote_block.h`, `rad_block_moe_fp8.h` and `rad_block_hc.h` attachment sites to consume the **same** policy snapshot in placement, declare, HyperConn and runtime; no zero-argument global policy functions. Thread the policy view only where necessary: `Geom` has no owned binding today, so test the actual declaration path. Keep `StepTicket` wire unchanged.
5. **Gate:** first-open, helper-off, second-helper-on refusal, cross-rank policy agreement, old-map isolation, same-group-reused-after-close tests pass. Report deleted code and changed patch anchors. **Do not proceed** if net new registry/dispatch code exceeds the deleted old path without an explicit rationale.

### S2 — move one HookState behind the binding and remove DSO lifecycle

1. Turn `HookState g` into per-open allocated transport state. Bind existing `Ledger`, `RankCompletion`, `HelperWatch`, `EngineRankCoordinator`, rank buffers, owner bitmap and async threads through that object. Convert free callbacks and background thread entry functions to member methods or single context-taking functions; do not invent a separate worker-controller class. Remove `HookLifetime` and its implicit register/interrupt/destroy path completely.
2. Make the plugin factory `open` acquire resources with existing RAII or a single unified cleanup routine. Fix early `fstat/mmap/pin/watch/registration` errors and restore failed-start state without stale handles. The exclusive helper claim is a simple no-policy guard, not a new mutable singleton. The second helper-on instance must not modify A.
3. Migrate **all** direct functions and internal `LiveRankOperations` invocations to the same explicit transport instance: `r3_moe_mailbox_device`, `owner_generation_{upload,ready}`, `close_admission`, `all_ranks_consumed`, `wait_drained`, `reopen_admission`, ticket/owner APIs. Remove old rank-only globals, do not keep aliases. Keep placement epoch/slot and engine-owner/session distinct.
4. Change `moe_mailbox_plugin.cpp` to return stable static rows at load. Have `moe_mailbox_ops.hip::mailbox_init` resolve the active generation **once** using upstream-supplied `tp_group` (prefer existing shared-slot or owner lookup, only one scoped non-owning map if necessary), validate rank/world/owner and create a tiny borrowed kernel instance. Give every affected `RadKernelInfo` a `fini` that retires it. Launch via `RadArgs::instance` only.
5. **Gate:** no DSO initializer side effects; standalone schema selftest works; injected errors at each acquisition leak nothing; bad/stale group fails before HIP access; old engine's kernel instance never sees new generation; no per-token lookup or extra sync. Source tests reject `HookState g`, `HookLifetime`, `g_plugin`, implicit `RTLD_DEFAULT` registration and old exports.

### S3 — consolidate clean and fault lifecycles

1. Inspect `Engine::serve`, `step_loop`, `fail_loop`, `Engine::drain_streams`, rank worker exits, and registry shutdown. Remove the shared `bc_moe_remote_shutdown` "always interrupt" path; create one **small** scoped stop flow with two outcomes, not two separate teardown implementations.
2. **Clean:** finish existing ledger admission, cancel only unused/unbound `ahead_`; let bound `current_` and `ahead_` complete, seal/host_done exactly once; wait existing terminal event proof and helper drain, then rank/tape/transfer/upload streams and helper threads; use normal registry instance `fini`, detach maps/callbacks, free transport. Do not use `barrier_->abort` during valid clean draining.
3. **Fault:** poison existing Ledger and rendezvous, notify waits, interrupt rank work, join/retire all peers and streams, then run the **same release tail** as clean. Preserve failed helper ID/session in error, never call a successful clean-drain proof or fallback to local experts.
4. **Partial start:** run that same release tail on failure. Ensure peer-published `scratch/flags/seq` buffers survive until all peers unwind and GPU readers are done. Reuse rad0017's `free_rank_buffers`, `free_failed_rank`, `bc_shared_slot_retire_owner`; update `r4d_ar_oneshot_2rank_exact.hip` only after an actual failing test. Preserve wake/fence/chained kernel code byte-for-byte unless test establishes a bug.
5. **Gate:** every injected failure returns without a leak or UAF, no peer hangs on timeout/fault, no double `fini`, clean issue-ahead drain success, helper death reports explicit cause, reused engine/group never sees prior pointers.

### S4 — apply/replay/build tests, code-reduction review

1. Compose the exact managed patch order; inspect overlap with `rad0004`, `rad0009`, `rad0016`–`rad0033`, `rad0049`; never alter archive patches. Confirm no second upstream attachment site where existing one suffices and no added generic module not reused twice.
2. Build touched core/libr3moe/collective targets with `nice -n 19`, one build at a time. Run existing CTest and Python patch tests and strict ABI/source/forbidden-term guards; verify archived patch replay is still identical. Distinguish skipped source tests from actual passes.
3. Produce a **measured deletion ledger**: production lines added/deleted, injected patch anchors and upstream touched files before/after, removed singleton owners, removed duplicate policy/cleanup bodies, new/internal C exports and all migrated consumers. If production net LOC increases, stop and submit the exact necessary safety exception plus alternatives rather than adding an unreviewed abstraction.
4. **Gate:** no hidden process-global policy, no new duplicate path, correct no-helper model, all card-free tests and build/ABI/patch gates green. Commit only coherent tested slices.

### S5 — short leased GPU proof on installed 1170.37.0

1. Use the existing host lease mechanism, one named GPU workload per short lease; preserve other agents' jobs and stop only owned PIDs. The installed runtime is used where appropriate, but code-under-test must be the rebuilt candidate, and every receipt records which binary was used. Do not mislabel an old release canary as a new refactor test.
2. **H1 code-object:** heterogeneous ranks choose executable code objects for their actual gfx1100/gfx1201 cards; no device-0-only check.
3. **H2 ordinal reversal:** reverse the visible rank/card order and prove `Engine::owner_`, `slot_base_`, native `Config.devices`, HIP stream ordinal and chosen kernel agree on every rank; no physical BDF/ordinal conflation.
4. **H3 rendezvous:** mixed-DSO same-group join; different-group isolation; timeout rollback/poison; no reuse until previous peers and GPU users retire.
5. **H4 span:** zero-width and ragged TP spans preserve declarations, scale tensor geometry, shard ownership and exactly-once calls.
6. **H5 short helper-on canary:** selected released config on the candidate: start, bound steps, clean stop and a second same-process Engine lifetime; helper-off control and (where feasible) injected helper-fault refusal. Keep source/model and numerical quality checks as required by RUM06; no broad extra sweep without evidence.
7. **Gate:** four separately named fresh hardware proofs with lease receipts, exit codes, backend/provenance, raw logs and a compact tracked summary. Hardware or quality regressions fail the gate.

### S6 — review, PR, plan/ledger closure

1. Re-read [DESIGN-REVIEW.md](DESIGN-REVIEW.md) and repeat the adversarial source/ABI/cleanup/duplication challenge against the actual diff. Resolve each critical finding with a failing-before/passing-after test.
2. Update [README.md](README.md) per existing RUM06 clause with source, commit, test and path. Preserve gap/partial labels until proof is present. Push the **existing branch** to **PR #1272**, do not create another PR; check CI and actual merge state, and do not misreport pending auto-merge.
3. Use Planning MCP to revise RUM06 and close RV4303 only when warranted; emit Ledger MCP unreleased event with change-class/files/technical summary/user summary/plan ID. If unavailable, do not manually edit authoritative Planning/Ledger state. Hand lead exact acceptance/closure text. No completion claim before proof and state reconciliation.

## Detailed Solution & Technical Design

The [design](ENGINE-GENERATION-DESIGN.md) is the source of truth for ownership and per-phase API. Important invariants:

- Engine owner/TP-group/session are validated at binding, while the helper's existing epoch/slot/parent ticket remain the *only* runtime placement/admission tokens.
- Single state owner. The Engine has a small opaque binding; DSO supplies the implementation. `Registry` already handles DSO lifetime and instance `fini`. The **only** allowed process-global helper state is a lightweight exclusive ownership gate, not callback/policy/maps/HookState.
- A normal stop **never** interrupts an otherwise healthy ledger; a fault stop poisons/wakes without claiming success; all borrowed pointers survive until proper unwind.
- The same immutable policy predicate drives placement, declaration, HyperConn and runtime. A model cannot run with policy from the prior Engine generation.
- No new dispatcher under the names `EngineExecutionState`, `RemoteExecutionStateRegistry`, `KernelLeaseManager` etc. unless the replacement demonstrably removes at least two older subsystems (currently there is no evidence it would).

## Code Samples & Guidance

```text
static loader: Registry::plugin("libr3moe")->handle, immutable kernels
Engine: native owner + claimed TP group -> RemoteBinding::open
           -> existing admission and declare -> existing kernel instance init/fini
clean: finish existing ledger -> settle/cancel ahead -> rank+helper drain
           -> normal Registry::fini_instances -> detach -> RAII release
fault: poison existing ledger/rv -> wake/unwind peers -> same release tail
```

Start verification using existing test entrypoints; commands below are examples, not claimed execution:

```sh
pytest -q tools/tests/patch/test_rum06_registry_lifetime.py \
  tools/tests/patch/test_vtp14_n1_rank_code_selection.py \
  tools/tests/patch/test_vtp14_n2_tp_ownership.py \
  tools/tests/patch/test_vtp14_n3_shared_rendezvous.py \
  tools/tests/patch/test_vtp14_weighted_twoshot_3rank.py
pytest -q tools/tests/patch/test_radiance_patch_packages.py -k rad0002
nice -n 19 cmake --build <existing-project-build> --target <focused-target> -j1
ctest --test-dir <existing-project-build> -R <focused-test> --output-on-failure
# Only after GPU lease admission, with the actual repo test command:
agent-lease acquire gpu --ttl 15m --est 10m --wait -- <named-hardware-test>
git diff --check
```

## Files

**Modify:** `engines/radiance/patches/rad0002_moe_remote_handoff/patch.py`; `engines/radiance/components/moe-remote/{moe_remote_hook.cpp,moe_remote_hook.h,moe_mailbox_plugin.cpp,moe_mailbox_ops.hip,CMakeLists.txt}`; existing relevant host fixtures and `tools/tests/patch/test_rum06_registry_lifetime.py`. Edit `engines/radiance/components/serving/{bc_engine_settings.h,bc_serving_manifest.cpp}` only if typed snapshot access genuinely requires it.

**Review/reuse, modify only with a red test:** `engines/radiance/components/moe-remote/{admission_ledger.h,admission_completion.h,engine_admission_gate.h,engine_rank_coordinator.h,owner_generation.h}`; `engines/radiance/patches/rad0017_vtp14_n3_shared_rendezvous/{patch.py,src/bc_ar_rendezvous_impl.h,src/bc_ar_lifecycle_impl.h}`; `rad0018`, `rad0019`, `rad0020`, `rad0025` and downstream TP/fence patches. Do not edit frozen/archive patches.

## Validation

| Gate | Concrete proof | Result now |
| --- | --- | --- |
| T0 loader | Static rows; handle-resolved factory; size/ABI refusal; standalone selftest | Open |
| T1 generation | A→detach→B; old maps/policy/instance unusable; reused native group handled | Open |
| T2 exclusivity | A active, concurrent helper-on B refused, A unaffected, C helper-off fine | Open |
| T3 scheduler | One parent ticket, issue ahead bound/unbound, terminal events and host completion exactly once | Partial prior |
| T4 fault | Helper death/timeout poison/wake, named failure, no false clean drain | Partial prior |
| T5 resource ownership | Each pre/post-publication failure injected, zero host/VRAM leaks, after peer unwind | Partial prior |
| T6 plugin/TP agreement | Cross-DSO group, zero-width/uneven rank, fence unchanged, exactly-once collective | Partial prior |
| T7 simplification | Deleted LOC > added LOC and fewer owned global paths; patch-anchor review | Open |
| H1–H4 GPU | Code object, ordinal reversal, rendezvous, span, each independently passed | Open |
| H5 canary | Candidate helper-on start/canary/clean stop/restart and helper-off control | Open |
| CI/closure | Exact patch replay, targeted build, required CI, Planning/Ledger state | Open |

**Previous release evidence does not close T0–T7 or H1–H5 for the refactor.** The initial narrow fixes on PR #1272 are already audited; this document is not evidence of new executable behavior.

## Effort & Risk

P0 / L. Highest risk is *ordering*, not number of lines: static loader before Engine bind; placement policy before declare; per-op init borrows a live context; successful stop must delay instance finalizers and transport free; failed startup must still unwind peer-visible addresses. Every migration slice must delete its superseded path and pass the regression gates before moving on. No over-generalized framework that makes later work harder.

## Standards

Follow `docs/reference/agents/WORKFLOW.md`, Tooling/patch ownership, repo-relative evidence and one-task worktree discipline. No absolute host names/paths, attribution lines or hidden compatibility shims in commits. Plan and review authority remains MCP; do not manually alter status/event files. GPU tests need actual leases. Use current Radiance 2.2 source, not outdated ABI43 line anchors.

## Acceptance Criteria

**All are required:**

1. Exactly one Engine/TP generation owns every callback, policy, map, direct mailbox/owner upload/ready operation and collective binding. No DSO static registration, no process-global remote-MoE mutable execution state.
2. Clean finish closes admission, cancels unbound tail only, completes all bound parents with exact per-rank/host/GPU proof, drains helper/peers and then detaches/frees. Fault stop wakes every waiter and retains published buffers through unwind, with a specific session/helper error.
3. Partial-init injected failure frees each rank-local allocation exactly once on its owning device and never frees a pointer peers can still read; repeated owner/group lifetimes and concurrent second helper-on refusal are proven.
4. Existing mixed-TP geometry, zero-width/tail declarations, cross-DSO TP grouping, chain/fence contract and numerical policy are preserved. All four named fresh hardware gates and representative candidate canary pass.
5. **Refactor benefit:** final production diff is net negative LOC by measured owner-source/patch footprint; duplicated mutable owners/dispatch/policy/cleanup code are removed, generic machinery reused, no new fallback/duplicate architecture. Exceptions require a separate explicit accepted rationale; not silently waived.
6. Existing PR #1272 passes required checks; RUM06 accepted/closed via Planning MCP with linked review and Ledger event, never manually fabricated.

## Notes

Prior audit commits: `090437657`, `22388b657`. Relevant reviewed release: bc-radiance 1170.37.0. Existing plan RV4303 and `docs/reviews/radiance-upstream-2026-11-18/patch-footprint.json` are required starting evidence; use exact current source at implementation time. This draft is deliberately **not** evidence that RUM06 is completed.
