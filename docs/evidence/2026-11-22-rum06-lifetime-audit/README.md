# RUM06 engine/helper lifetime audit — 2026-11-22

Base: released Radiance 2.2 / bc-radiance 1170.37.0, main at 618d7447e.
Scope: RUM06 acceptance and review RV4303. This audit does **not** mark
RUM06 complete. Prior release canaries do not prove helper lifecycle reuse.

## Clause-by-clause disposition

| Acceptance clause | Result | Source and proof / exact gap |
| --- | --- | --- |
| Engine/TP-generation ownership for callbacks, maps and policy | **Gap** | `engines/radiance/patches/rad0002_moe_remote_handoff/patch.py` still defines process-global `g_plugin`, `g_layer_maps` and `g_mu`. This slice now refuses replacement of a live plugin, clears retired maps on unregister, and avoids first-generation static policy caches, but it does not bind callbacks to a typed Engine generation. `tools/tests/patch/test_rum06_registry_lifetime.py` covers the narrow changes. |
| Direct mailbox/device and owner-generation upload/ready are engine-bound | **Gap** | `engines/radiance/components/moe-remote/moe_remote_hook.cpp` `r3_moe_mailbox_device`, `r3_moe_owner_generation_upload`, `r3_moe_owner_generation_ready` all resolve process-global `g` by rank and lack an engine identity argument. Need explicit bind/open and stop/detach with a validated generation token. |
| Clean stop closes admission, cancels unbound tail, drains bound work before teardown | **Partial** | `rad0002` `BcMoeRemoteAdmission::finish`, `bind_ahead`, `wait_finished` and `Engine::serve` implement the intended scheduling order; `engines/radiance/tests/admission_bridge_test.cpp` tests bound issue-ahead behavior. However `moe_remote_hook.cpp::close_mailbox` still calls `g.ledger.interrupt()` from a DSO destructor, and there is no explicit Engine-owned detach after a successful drain. |
| Faults unblock peers and prevent stale owner/address reuse | **Partial** | `rad0017_vtp14_n3_shared_rendezvous/src/bc_ar_rendezvous_impl.h` marks failed rendezvous and wakes waiters; `bc_ar_lifecycle_impl.h` holds published buffers until group unwind. Existing RUM06 checkpoints document tests. A full Engine-generation reuse and injected-helper-death fixture is still required for the direct API and callback surfaces. |
| Partial-init rank resources freed only after peer unwind | **Partial** | `rad0017` owned `bc_ar_lifecycle_impl.h` and the one-shot IPC patch contain pre-publication ownership cleanup and failed-group lifetime handling. Prior RUM06 evidence cites the 23-case host fault fixture. Fresh post-publication failure/reuse evidence for every partial-init path is missing. |
| No process-global helper policy or DSO constructor registration | **Gap** | `moe_remote_hook.cpp::HookState g`, `HookLifetime` constructor/destructor remain production globals. Removing static one-time policy caches in `rad0002` is not a full replacement. Requires Engine-owned state and explicit lifecycle hooks. |
| Unsupported second helper/model instance refuses cleanly | **Partial** | `bc_moe_remote_register` now refuses replacing a live plugin and unregister clears layer maps, but the direct device and owner-generation API still has no distinct Engine/session ownership. A concurrent-model integration refusal test remains missing. |
| Rank ownership, code-object, ordinal, rendezvous and span tests on mixed TP hardware | **Gap (new proof)** | `rad0016` and `rad0017` include per-rank kernel selection and grouped rendezvous; existing patch fixture suites pass, with pinned-source tests skipped when source is unavailable. The four separately requested fresh hardware tests were not executed by this slice. |
| Selected production layout starts, canaries and stops on released 2.2 | **Met (prior release)** | `docs/planning/completed/patching-radiance-upstream/RUM09.md` and its Gate E release evidence establish the installed bc-radiance 1170.37.0 roster; all 18 shipped names have successful start/canary/stop entries. This proof is not a remote lifecycle audit. |

## Narrow changes and reproducible checks

- `moe_mailbox_plugin.cpp` reports availability from the **current** open helper generation, rather than pinning the first DSO lookup's enabled state; `moe_mailbox_lifetime_test.cpp` runs disabled → enabled → disabled → enabled.
- `rad0002` rejects registration while a plugin is live, clears stale maps on unregister, and re-reads generation-selected policy instead of caching the first instance.
- `tools/tests/patch/test_rum06_registry_lifetime.py`: 3 passed (failure reproduced before each code fix).
- `test_radiance_patch_packages.py -k rad0002`: 5 passed.
- `test_vtp14_n1_rank_code_selection.py`: 4 passed, 1 skipped.
- `test_vtp14_n2_tp_ownership.py`: 4 passed, 1 skipped.
- `test_vtp14_n3_shared_rendezvous.py`: 2 passed, 2 skipped.
- `test_vtp14_weighted_twoshot_3rank.py`: 2 passed, 1 skipped.
- Card-free CMake target `moe_mailbox_lifetime_test`: compiled with the released ABI headers; 1/1 CTest passed.
- `git diff --check`: clean.

## RUM06 closure gate

**Open.** The minimum coherent remaining implementation is an opaque
Engine-generation binding in the managed `rad0002` extension and libr3moe
`moe_remote_hook.cpp`, including all direct entry points and an explicit
close/cancel/drain/detach lifecycle. Tests must cover repeated generations,
concurrent second-engine refusal, fault wake, partial initialization and
published-pointer lifetime. Then run the four named mixed-TP hardware tests
under short GPU leases. No plan completion is justified while gaps remain.

## Design handoff (not implementation proof)

- [Engine-generation ownership design](ENGINE-GENERATION-DESIGN.md)
- [Implementation plan draft (Planning MCP handoff)](IMPLEMENTATION-PLAN-DRAFT.md)
- [Adversarial design review](DESIGN-REVIEW.md)
- [Mandatory code reduction and generic reuse gates](SIMPLIFICATION-GATES.md)

These artifacts propose future code changes. No RUM06 lifecycle or hardware acceptance gap has been closed by the design documents.
