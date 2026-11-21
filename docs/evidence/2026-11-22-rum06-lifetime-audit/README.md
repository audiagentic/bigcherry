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
- `tools/tests/patch/test_rum06_registry_lifetime.py` (3 pass) encodes the **old** process-global contract; replace with real behavior test when S1 lands.
- Prior mixed rank/TP fixtures and installed Gate E tests do not prove a new generation binding.
- Do not silently complete RUM06 on PR merge; RV4303 remains open until proof.

## S1 implementation sequencing blocker — lead decision required

The approved S1 must remove `bc_moe_remote_register`, `g_plugin` and the unscoped callback route **without** leaving a compatibility path, while the approved S2 is assigned removal of DSO constructor registration. These cannot be independently green in that literal order: `moe_remote_hook.cpp::HookLifetime` constructs at dlopen and invokes `register_mailbox()`, which resolves `bc_moe_remote_register` and registers callbacks; upstream `Engine::load_plugins` claims the native owner/group **after** plugin loading. At constructor time there is no Engine generation to receive the new context.

Approve either (a) move removal of DSO callback registration and the exact-handle explicit factory bind into **S1**, leaving per-generation HookState ownership and full teardown for S2, or (b) treat S1+S2 as one atomic source/ABI cutover with one final green gate. A temporary global pending registry, old ABI shim, TLS-selected generation or plugin ABI change would violate the decision. No S1 production code was changed or tests claimed. This is a slice-boundary decision, not a request to narrow any RUM06 acceptance clause.

## Slice results

| Slice | Commit | Deleted/reused | Tests | Hot-path cost |
| --- | --- | --- | --- | --- |
| S1 | Blocked: slice-boundary decision | None | No new test | None |
| S2 | Open | — | — | — |
| S3 | Open | — | — | — |
| S4 | Open | — | — | — |
| Mixed-TP | Open | — | — | — |
