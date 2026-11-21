# RUM06 — minimal engine-generation ownership design

**Status:** proposed, reviewed against source, not implemented or accepted. **Authority:** existing [RUM06](../../planning/active/patching-radiance-upstream/RUM06.md) and [RV4303](../../planning/active/patching-radiance-upstream/reviews/RUM06/RV4303.md). Related: [implementation plan draft](IMPLEMENTATION-PLAN-DRAFT.md), [design review](DESIGN-REVIEW.md), [audit](README.md).

## Design decision: replace, do not extend

The work is justified **only if the final owned production code is simpler and smaller**. Do not layer a second lifecycle on the current static hook. Remove the old globals, singleton callback dispatcher, duplicate policy readers, DSO registration, and map replay; replace them with one small Engine-owned binding backed by the **existing** admission, TP-group, registry and device-instance mechanisms. Refactor all internal callers in the same change, no compatibility shims or fallback code. No changes to upstream's public plugin ABI and no new generic framework whose only user is remote MoE.

BigCherry-owned, not upstream: `rad0002` injects `abi/bc_moe_remote.h`, `core/place/bc_moe_remote.cpp`, block/placement/engine hooks; `libr3moe` implements helper transport; `rad0017` extends native shared collectives. Upstream `Engine`, `Registry` and `RadKernelInfo` already provide ownership scopes. The design's **only new concept** is an explicit per-Engine *remote binding*. Use existing generic mechanisms; do not create a second plugin framework.

## Actual source constraints

1. `vendor/radiance/core/plugin/loader.cpp` loads each DSO with `RTLD_LOCAL` and enumerates `rad_kernel_count/at` **before** `Engine::load_plugins` calls `rad_engine_claim`. A DSO constructor or a dynamic kernel row count cannot be the binding point. `Registry::plugin("libr3moe")->handle` already exposes the DSO handle for one narrow `dlsym` factory call.
2. `vendor/radiance/core/engine_bringup.cpp` claims `slot_base_`, passes it to `reg_->set_tp_group(base)`, then declares. Bind after claim and before declaration. The existing immutable Engine owner (`owner_`) and TP group (`slot_base_`) are separate from the helper's placement epoch and slot.
3. `vendor/radiance/core/plugin/select.cpp` injects `tp_group` in every kernel init and calls `RadKernelInfo::fini` on registered instances. Reuse **that** generic lease/cleanup lifecycle, not a new callback scheduler. `rad_build_slot` and `rad_slot` disambiguate different engines; rank alone is insufficient.
4. `rad0002_moe_remote_handoff/patch.py` currently contains global `g_plugin`, `g_mu`, `g_layer_maps`, `g_burst_active` and an implicit TLS rank ticket. The policy is repeatedly read through `bc_serving_remote_settings` rather than frozen per admitted Engine. The injected C ABI lacks context.
5. `moe_remote_hook.cpp` contains `HookState g`, `HookLifetime g_lifetime`, `register_mailbox` and `close_mailbox`. Direct device, upload, ready and reusable placement close/reopen APIs bypass the callback registry. `close_mailbox` currently interrupts the ledger, invalid for clean drain; some early `register_mailbox` returns do not release fd/mmap.
6. `moe_mailbox_plugin.cpp` counts kernel rows conditionally and `mailbox_init` only receives rank/world. Keep static kernel metadata always visible; fail closed in instance init without an admitted binding.
7. `rad0017/.../bc_ar_rendezvous_impl.h` already owns `rv_for(tp_group)`; `bc_ar_lifecycle_impl.h` has rank buffer release and deferred cleanup across shared slots, backed by `rad_bc_shared_slot` and `bc_shared_slot_retire_owner`. Keep it. A new cross-DSO rendezvous registry would duplicate working code.
8. `vendor/radiance/core/build/startup.cpp::rad_engine_claim` already prevents a second live engine **for the same non-multi-instance architecture plugin** and reserves numbered slots. It does **not** prevent two different architecture plugins sharing a single remote helper. Keep a **single small exclusive helper-owner claim** only for this residual case, not a new global state collection.

The task worktree lacks a local `vendor/radiance` checkout; source observations above were made against the canonical working source mirror. Reconfirm source pin/anchors before implementation.

## One ownership graph, minimal additions

```text
Radiance Engine (native owner_, slot_base_, Plugin Registry)
    ├── existing BcMoeRemoteAdmission        [one coordinator; reused]
    └── RemoteBinding                        [ONE new move-only RAII value]
         ├── snapshot Policy                 [the existing typed manifest fields]
         ├── callback table {instance, funcs}[existing hooks made context-aware]
         ├── layer maps / burst state        [moved, never duplicated]
         └── opaque HookState*               [libr3moe's existing state moved off static g]
              ├── existing Ledger + RankCompletion
              ├── existing HelperWatch + epoch coordinator
              └── existing per-rank GPU allocations/mmap/events
                     ↑ borrowed by existing kernel init/fini instances
libr3moe static .so: immutable kernel metadata, no active helper state
```

There must be **one** real state owner. The engine `RemoteBinding` can be a small typed C++ wrapper with a pointer, policy and callbacks; transport `HookState` stays opaque and is freed by its DSO. Do **not** introduce a second `EngineExecutionState` class, service manager, lease manager, typed epoch service, owner-map cache or transient callback registry.

**Lookup at init only.** The existing kernel-init signature supplies a native `tp_group` but not an Engine pointer. First investigate whether the currently exported `rad_bc_shared_slot`/owner scope can safely supply a borrower slot. If not, permit **one non-owning group→binding lookup** in the core remote extension; it is a *lookup only*, never owns helper state, callbacks, resources or policy. Always validate the Engine owner and group, retire it before freeing transport, and use each kernel instance's existing `fini` for borrowed-lifetime release. Do not introduce a general-purpose registry or a new scheme for all plugins.

## Explicit internal contract

A single size-checked context-aware callback table replaces the existing stateless `BcMoeRemotePlugin` table; include `void* instance` and pass it to every callback. The engine owns the resulting opaque binding; the DSO implements `open(binding inputs)` and `close(instance)` exports resolved from the already loaded `Plugin::handle`. Bind inputs include native Engine owner/group/world, immutable remote policy and admitted helper model/session identity. Exact C fields to be settled with a compile/red test before code changes; **do not** add both old and new variants.

The existing `StepTicket` stays 16 bytes and retains only its present placement epoch/slot and parent ticket ID. **Never** put an Engine pointer into a GPU ticket. A rank's TLS may carry the ticket during the existing rank barrier, provided the rank's associated `RemoteBinding` is explicit, validated and cleared on worker reuse. Eliminate TLS as a way of selecting policy or transport.

The caller partitions are:

| Phase | Single source | Remove redundant path |
| --- | --- | --- |
| Plugin discovery | Upstream `Registry` + stable `rad_kernel_count` | `HookLifetime` constructor registration, live-count kernel gating, alternate loader |
| Native policy | Bound snapshot of existing `BcEngineSettings` | `RemotePolicy remote_policy()` repeated settings lookups and function-static policy cache |
| Placement / graph declare | Engine-owned policy and map, `rad_build_slot` | `g_layer_maps`, plugin-registry replay and implicit scalar policy queries |
| Runtime dispatch | Existing `BcMoeRemoteAdmission` + context-aware callbacks | `g_plugin`, `g_mu`, copied callback-function getter blocks, second admission gate |
| Mailbox kernels | Existing `RadKernelInfo::init/fini`, `tp_group`, `RadArgs::instance` | Rank-only direct global `r3_moe_mailbox_device` |
| Owner transition | Existing `Ledger`, `EngineRankCoordinator`, owner epoch/slot | Free functions silently resolving `HookState g` |
| Collective peer ownership | Existing `rv_for(tp_group)` + shared-slot retire | Any per-DSO or per-rank duplicate rendezvous |

Build-time `BcMoeRemote` and HyperConn must use **the same snapshot**. Rewriting only the engine callback table while leaving `bc_moe_remote_wanted(T)` and `bc_moe_remote_hc_open(...)` resolving global policy is **not acceptable**; pass a small policy view from the already available `RadBuildCtx`/Engine declaration scope. Prefer one shared predicate in the existing BigCherry block header rather than copies. An architecture `Geom` currently has no remote-generation field; thread an explicit view only where required, not through every unrelated architecture.

## Lifecycle (no second controller)

```text
unbound -> opening -> active -> closing -> drained -> detached
                    \-> fault -> unwinding ----> detached
       \-> open-failed -> rollback -> unbound
```

**Open:** Engine claims native TP group first; ensure exclusive helper-on claim before resource acquisition; validate session/model and policy; acquire fds/mmap/pinned memory and helper watch; initialize existing ledger; publish context; spawn existing coordinator last. RAII rollback each acquired resource on failure; no double registration and no fd/mmap leaks. Helper-off creates no helper transport.

**Clean stop:** finish admission first, do not select a new parent, cancel **only unbound** ahead reservation; complete any bound ahead/current with normal rank seal and host terminal proof, use existing Ledger wait + helper drain; join rank/tape/transfer/upload streams, event readers and background threads. Let **existing** Registry fini release each mailbox instance, then detach callbacks/group and free rank/shared memory. No ledger interrupt and no barrier abort until clean wait has ended; fail explicitly on deadline rather than claiming drained.

**Fault stop:** preserve specific failed helper/session and error cause; poison ledger and collective, wake peers and interrupted waits; join/unwind ranks, tape/streams and published peer-pointer readers, then fini/detach/free. Do not relabel a helper failure as generic OOM, reset a live group's `rv`, or fall back to local expert compute.

**Placement close/reopen** is distinct from terminal Engine close. Reuse its existing `R3MoeDrainSnapshot` and the two owner-generation slots; no new epoch or separate completion system.

**Second helper-on Engine:** fail before weight declare or GPU allocation; first Engine's buffers, maps, policy and admission remain untouched. After full detach/release, a fresh Engine may open, even when the native slot/group number is reused. Helper-off/independent native Engine remains valid.

## Code-reduction and generic-reuse contract (mandatory)

Measured starting line counts at task head `22388b657` (physical `wc -l`, **not** semantic complexity; record final diff and maintained patch anchors too):

| Owned baseline file | Lines |
| --- | ---: |
| `rad0002_moe_remote_handoff/patch.py` | 1,205 |
| `moe_remote_hook.cpp` | 1,336 |
| `moe_remote_hook.h` | 173 |
| `moe_mailbox_plugin.cpp` | 45 |
| `moe_mailbox_ops.hip` | 1,331 |
| `rad0017/.../bc_ar_lifecycle_impl.h` | 478 |

**Delete or consolidate, don't wrap:** `g_plugin/g_mu/g_layer_maps` and callback getter boilerplate, global burst state, duplicate policy fetches/functions, map replay once obsolete, `HookState g`, `HookLifetime`, global direct-API call sites, rank-only mailbox resolve, repeated partial-init releases and stop/fault conflation. Count removed symbols and lines, and inspect generated patch footprint (`patch-footprint.json`) so the upstream anchors do not expand gratuitously.

**Net change gate:** for touched *production* sources and injected patch payloads (excluding tests/docs), **deleted LOC should exceed added LOC** for the completed migration. Also report: (a) before/after active mutable singletons, (b) before/after independent lifecycle owners and lookups, (c) before/after total patch anchors/modified upstream files, (d) duplicated predicate/cleanup bodies eliminated, (e) resulting public/internal ABI exports and stale references. A useful generic must serve **at least two actual call sites in this migration or an existing unrelated subsystem**, and replace their duplication; otherwise keep it local and smaller. If safety forces a positive net LOC delta, provide the exact unavoidable new invariant and net cost in the review, eliminate other duplication first, and obtain an explicit exception rather than hiding the increase in a generic helper. No handwritten model/geometry constants.

**Do not force line reductions by deleting tests, diagnostic failure details, safety invariants, existing supported modes, fences or stream synchronisation.** The simplification is fewer state owners and fewer code paths with equivalent behavior. In particular, do not split `HookState` into many new classes or refactor all collective backends just to produce generic code.

## Invariants / rejection conditions

- One native owner + one TP group + one exact helper session bound per active remote generation; rank 0 is not a valid owner key.
- A close cannot overtake borrowed callback/kernel instances, captured tape or published peer pointers. No group reuse until all peer joins/stream drains complete.
- Existing Ledger is sole reserve/bind/cancel/host_done authority. No new task queue or placement epoch service.
- Preserve AQL acquire/doorbell/wake and chained fences, mixed-rank span ownership, zero-width/uneven declarations and exactly-once collectives.
- No remote helper used without an admitted native/BC policy and exact bitmap/session proof. No plugin DSO constructor registration.
- Refuse any proposal that adds a second singleton, copies the lifecycle into a new framework, creates a global fallback, or increases patch surface without a measured removal.

The [plan draft](IMPLEMENTATION-PLAN-DRAFT.md) maps these constraints into small dependency-ordered slices with stop conditions, proof gates and a deletion ledger. The [review](DESIGN-REVIEW.md) challenges this proposal; none of the gates is currently green merely because it is documented.
