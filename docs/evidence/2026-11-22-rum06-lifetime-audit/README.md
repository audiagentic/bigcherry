# RUM06 remote-MoE lifetime audit — 2026-11-22

**Authority:** existing RUM06 / RV4303; approved internal ABI direction, not plan completion.
**Baseline:** Radiance 2.2 / ABI44, released 1170.37.0. Actual source and test receipts must be pinned for each implementation slice.

## Current acceptance disposition

| RUM06 clause | State | What remains to prove |
| --- | --- | --- |
| Engine/TP-generation callback, layer map and policy ownership | Gap | Per-Engine binding, immutable typed policy and no process-global registry; reject concurrent helper ownership |
| Direct mailbox/device, generation upload and ready | Gap | Exact group/owner/session identity, stale/wrong-group refusal before HIP |
| Clean stop and bound-ahead drain | Partial | Finish admission, cancel only unused tail; drain parent/rank/device/helper before detach, never interrupt a healthy ledger |
| Fault wake and stale-owner refusal | Partial | Wake all ranks, preserve failure cause, process can safely open a later generation |
| Partial initialization after published pointers | Partial | Peer unwind before GPU free; no leaks, no double free, no reuse of failed shared slot |
| No global helper policy or DSO lifetime constructor | Gap | Delete `HookState g`, `HookLifetime`, `g_plugin/g_mu/g_layer_maps/g_burst_active` |
| Second concurrent helper-on Engine refusal | Partial | Refuse before mutating first Engine while allowing helper-off coexistence |
| Four mixed TP hardware tests | Gap | Fresh code-object, card ordinal, cross-group rendezvous and span tests on built PR branch |
| Installed release smoke | Prior proof only | Release 1170.37.0 roster; not proof of the new lifecycle |

## Approved implementation design (single note)

**Decision:** replace only BigCherry's `abi/bc_moe_remote.h` and consumers, not upstream Radiance `rad_abi.h`, `RadKernelInfo::init/fini` or plugin layout fingerprint. Exactly one Engine-owned remote generation binding holds a frozen typed policy, size-checked *context-aware* callback table, layer maps and burst state; libr3moe's opaque per-binding `HookState` owns existing Ledger, rank events, HelperWatch and GPU buffers. The native Engine owner and claimed TP group are separate from the helper session, placement epoch/slot and 16-byte StepTicket.

**Discovery / lifetime:** upstream Registry discovers **static** plugin kernel rows before Engine claims its TP group. Bind after native `rad_engine_claim` and before declaration, resolving the factory on the *already loaded* `Registry::plugin("libr3moe")->handle` (RTLD_LOCAL). Kernel instances receive `tp_group` at `init`, borrow the owned generation and release it at upstream `fini`. Reuse `rad_bc_shared_slot` and `rad0017` rendezvous/deferred free. If shared slots cannot identify the borrowing Engine, at most one *non-owning* group→binding lookup is permitted **at init only**, erased before numeric group reuse. One minimal exclusive helper-on owner claim is permitted; native arch claim does not protect against different architecture plugins.

**No alternate implementation:** delete the existing DSO constructor/destructor, global callback registration, per-call manifest reads, map replay, TLS-selected ownership, duplicate placement or cleanup controller; no compatibility shim, additional policy parser, registry, loader or collective. Do not call plugin callbacks while holding a layer map mutex. Keep helper-off graphs without mailbox ops. HyperConn, placement, block declaration and runtime must ask the same bound policy predicate. Rank ticket can remain a value **only** if its associated Engine binding is explicit and validated; never use TLS to select policy or transport.

**Stop ordering:** on success: close admission, cancel only unused unbound ahead tail, complete bound current/ahead parent and rank seal/host_done, wait helper/rank proof, join worker/snapshot/placement threads, drain native Engine GPU streams, invoke per-instance `fini`, detach and free transport, then release exclusive helper claim. On fault: poison existing Ledger + TP group, unblock peers, join/drain and use the same release tail. Failed rank publication must not free peer-visible device pointers until all peers unwind. Clean success must not call `Ledger::interrupt` before proof.

**Performance constraint:** no new lock, allocation or group lookup on normal per-step decode/prefill; all changes to state ownership happen at Engine bind/declaration/kernel init/cleanup. Each slice records removed code, tests, hot-path cost and commit. Do not claim latency improvement without uninstrumented measurement.

## Approved execution slices

| Slice | Scope | Red/green proof required |
| --- | --- | --- |
| S1 | Engine-owned binding; explicit policy, maps, callback context; delete four engine globals | Actual Engine+callback behavior, sequential/concurrent policies and graph/placement parity |
| S2 | Remove `HookState g` and `HookLifetime`; explicit close/cancel/drain/detach | Clean stop, binding borrow lifetime, no DSO-triggered startup/free |
| S3 | Bind device mailbox, owner upload/ready to exact generation | Wrong group/owner/world and stale instance rejected before HIP |
| S4 | Fault, partial-init, second generation/concurrent refusal | Deterministic fault-injection/peer wake/UAF/no-leak tests |

**After S4:** build branch with nice 19, one build at a time; four named mixed-TP GPU tests on short lease; compare installed 1170.37.0 control. Reconcile every acceptance clause from fresh evidence, and leave Planning/Ledger state to the lead.

## Existing evidence and accepted limits

- Narrow prior changes: callbacks refuse second register, clear replay maps on unregister and re-read typed settings instead of freezing first generation. This is **not** the owned lifetime.
- CTest `r3_moe_mailbox_lifetime_test` confirms static rows through disabled → enabled → disabled → enabled. It must remain registered in both CMake and `component.toml`.
- `test_rum06_registry_lifetime.py` (old process-global contract) is deleted; `engines/radiance/tests/moe_remote_generation_test.cpp` and `admission_bridge_test.cpp` replace it (run by `test_radiance_patch_packages.py`).
- Prior mixed rank/TP fixtures and installed Gate E tests do not prove a new generation binding.
- Do not silently complete RUM06 on PR merge; RV4303 remains open until proof.

## S1 sequencing decision (resolved)

The lead chose option (a): S1 removes `bc_moe_remote_register`, `g_plugin`, DSO constructor registration and the unscoped callback route together, and binds through an explicit factory on the already-loaded plugin handle; per-generation HookState ownership and full teardown stay in S2. Implemented as: `bc_moe_remote_open(handle, slot_base, ranks, limit)` (after `rad_engine_claim`, before declare) finds `bc_moe_remote_plugin_open` with `dlsym` on `Registry::plugin("libr3moe")->handle`; the Engine's `BcMoeRemoteAdmission` owns the opaque generation; consumers hold it through `Geom::bc_remote` (found once by `bc_moe_remote_lookup(rad_build_slot(ctx))` in `geom_from`), `Program::ctx` (planner, bring-up) and `Ctx` (rank thread hands generation and ticket in; no thread-local). The slot table holds non-owning pointers, is written only on open/close and read at init time. Publication order: refuse a bound slot before the factory runs, run the factory, publish each slot by compare-and-swap from empty (release); close clears the slots before the plugin close; a closed generation stays allocated until destroy and refuses every call with `BC_MOE_REMOTE_STALE`.

## Test inventory added by S1

- `engines/radiance/tests/moe_remote_generation_test.cpp` (run by `test_rum06_generation_lifetime_on_the_generated_engine_code`) and `admission_bridge_test.cpp` (run by `test_admission_bridge_preserves_parent_across_issue_ahead`): the generation on the generated engine code.
- `engines/radiance/components/moe-remote/moe_factory_bind_test.cpp` (`r3_moe_factory_bind_test`, registered in CMake and `component.toml`): dlopens the built libr3moe, resolves `bc_moe_remote_plugin_open` on its handle, drives `bc_moe_remote_open` through the engine's generation code. Host-only it proves the factory is exported, a refused open (no helper segment) binds nothing, bad arguments are refused, and a plugin without the factory is ABSENT. The plugin's own BUSY (second engine while one holds the helper) and a successful open need a live helper and a card (card smoke).
- Fixture `engines/radiance/tests/fixtures/b6b8751` (exact vendor 2.2.0 copies of the files rad0002/rad0004 edit; the 1.14.0 fixture predates the slot model). Consumed only by `tools/tests/patch/test_radiance_patch_packages.py` through `FIXTURE_PIN`: rad0002/rad0004 apply/idempotence/anchor-failure/pinned-blob tests, the thin-call rule, the policy-gate, fused-choice, admission-bridge, generation-lifetime and plugin-struct-mirror tests. Refresh it with each pin bump (copy the same files from the new pin and recompute `PIN_BLOBS`).

## Slice results

| Slice | Commit | Deleted/reused | Tests | Hot-path cost |
| --- | --- | --- | --- | --- |
| S1 | Implemented (see S1 report) | `g_plugin`, `g_mu`, `g_layer_maps`, `g_burst_active`, `bc_moe_remote_register`, `HookLifetime` ctor registration, `bc_moe_remote_poll`, TLS ticket + `rank_ticket/current_ticket` | `moe_remote_generation_test` (bind/unbind x50, stale-after-reclaim, foreign/unclaimed slots, per-generation maps/bursts, concurrent lifetimes, contention), `admission_bridge_test` on a real generation, patch-package tests on a 2.2.0 fixture | Removes a mutex lock/unlock from every admit/bind/cancel/host_done/rank_done/before_pass and the per-query manifest read from every `wanted`; adds one acquire load per call; no new lock or allocation |
| S2 | Open | — | — | — |
| S3 | Open | — | — | — |
| S4 | Open | — | — | — |
| Mixed-TP | Open | — | — | — |
