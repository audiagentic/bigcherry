# RUM06 design review — safety, duplication and long-term simplicity

**Review type:** source-based design critique of [ENGINE-GENERATION-DESIGN.md](ENGINE-GENERATION-DESIGN.md) and [IMPLEMENTATION-PLAN-DRAFT.md](IMPLEMENTATION-PLAN-DRAFT.md), **not** an independent human review or an executable acceptance run. **Finding status:** proposed fixes, unverified until red/green tests. Reviewed source at the task's pinned BigCherry worktree plus canonical Radiance 2.2 mirror; canonical vendor source is not checked out inside this task worktree. Original [RV4303](../../planning/active/patching-radiance-upstream/reviews/RUM06/RV4303.md) remains open for authoritative review state.

## Verdict

**Design is suitable for implementation only with the following constraints:** eliminate old code in each slice, use the existing generic engine/group/instance lifecycle rather than introducing a new framework, and prove clean shutdown and every borrowed-pointer lifetime with tests. A one-line context pointer added to `g_plugin` while `HookState g` and global policy remain is an **unacceptable partial migration**.

The minimum durable abstraction is **one Engine-owned RemoteBinding** and the existing HookState as its opaque implementation. An exclusive helper-on gate and one temporary group→binding lookup at kernel init may be necessary; they must remain **non-owning** and be justified against shared-slot reuse. No new class hierarchy, global service, owner-index cache, plugin loader, epoch controller or scheduler.

## Pass A — correctness and ABI ordering

| Severity | Finding / source | Consequence | Required proof or correction |
| --- | --- | --- | --- |
| **Critical** | `vendor/radiance/core/plugin/loader.cpp::Registry::commit_` enumerates `rad_kernel_count` **before** `Engine::load_plugins` claims `slot_base_`; current `moe_mailbox_plugin.cpp::live_count()` depends on `r3_moe_mailbox_enabled`. | Removing DSO constructor while keeping live kernel enumeration makes all helper rows disappear at load. | Static row table/count; helper-off models must declare no mailbox ops, and an accidental init must explicitly refuse. Test whole loader/selector, not just call the exported count. |
| **Critical** | Current `moe_remote_hook.cpp::HookLifetime` calls `register_mailbox` on dlopen and `close_mailbox` on dlclose. `close_mailbox` immediately calls `g.ledger.interrupt()`. | A clean stop invalidates its own event/drain proof; teardown depends on plugin unload order. | Eliminate DSO owner; bind after `rad_engine_claim`; use two transitions with shared release tail. Verify actual `Engine::serve` early returns and destructor order. |
| **Critical** | `moe_mailbox_ops.hip::mailbox_init` takes rank/world, while direct `r3_moe_mailbox_device`, `owner_generation_upload/ready` inspect `HookState g`. | A valid-looking rank selects the previous Engine's GPU allocations on same-process restart. | Inject `tp_group` at `Registry::init_instance` and use explicit generation for every direct call; check wrong owner, group, world, epoch/session and stale instance. |
| **High** | `RadKernelInfo::init` is an upstream fixed signature: parameters include `tp_group`, not Engine pointer. `Registry::plugin(handle)` exists and per-row `fini` is used by `Registry::fini_instances`. | A new plugin loader, generic context registry or changed upstream ABI duplicates supported mechanisms. | Factory via exact `Plugin::handle`; one nonowning group lookup only if shared-slot borrowing is insufficient; no hot-path lookup, instance fini releases borrowing. |
| **High** | `core/build/startup.cpp::rad_engine_claim` rejects simultaneous engines using the *same* non-multi-instance architecture; it does not reserve remote helper ownership across distinct architectures. | Claiming native Engine safeguards suffice would allow two helper users to race. | Retain exactly one exclusive-live **remote** gate; second helper-on refusal precedes any helper mutation. Do not modify `rad_arch_multi_instance`. |
| **High** | `bc_serving_manifest.cpp::bc_serving_engine_settings` reads a shared `g_engine_settings`; `rad0002::remote_policy()` repeatedly fetches current settings. | Two sequential starts with different policy can affect live runtime, or a second model can change settings before it is refused. | Capture typed settings **once** into Engine-owned snapshot before exposing another generation; verify immutable owned strings/lifetime, not pointers into a mutable serving manifest. Prove the second refusal leaves A's policy unchanged. |
| **High** | `rad0002` calls `g_plugin.layer_map` while holding `g_mu`; map replay also holds the lock across a callback; `libr3moe` may wait on helper map publication. | Moving to an Engine-specific mutex but still calling callbacks under it risks deadlock and makes shutdown cancellation hard. | Remove map replay if binding precedes declare; store immutable maps under appropriate phase ownership, call helper without holding the map lock, and document concurrent declare/seal visibility. |
| **High** | `rad0002` has zero-argument global `bc_moe_remote_wanted/shadow/mask_only` used by *both* placement and HyperConn. | Only replacing callback table still leaks a global policy path and could cause helper rows to be counted differently by two ranks. | Explicit `PolicyView` passed to both BigCherry-owned block and HyperConn at declaration and step; common predicate; no global getter fallback. |
| **Medium** | `rad0009` supports typed manifest identity; `bc_serving_remote_settings` currently weak when rad0009 is absent. | Removing the weak path incorrectly can break intentionally independent card-free patch fixtures. | Retain missing-manifest **refusal/off** semantics in a single bind boundary, but do not add a compatibility routing path or fake live settings. Update tests and test fakes atomically. |

## Pass B — concurrency, clean stop and memory safety

| Severity | Finding / source | Consequence | Required proof or correction |
| --- | --- | --- | --- |
| **Critical** | `Engine::serve` patched by rad0002 currently uses `bc_moe_remote_finish_admission` / `wait_finished`, and fault path `bc_moe_remote_shutdown` before `barrier_->abort`. `Ledger::finish_admission` is terminal, unlike placement `close_admission/reopen`. | A bound issue-ahead parent can be abandoned or a healthy clean stop can be poisoned by the generic shutdown call. | Exact timeline test: issued-ahead *bound* parent must finish/seal/host_done; only unbound tail can cancel; clean exit skips ledger.interrupt and premature barrier abort; fault exit wakes all. |
| **Critical** | `moe_remote_hook.cpp::LiveRankOperations` and placement/snapshot threads use free functions reading `g`; the DSO currently closes watcher, events and GPU rank buffers during unload. | Move to object without join discipline creates use-after-free across background callbacks, captured tape and uploaded owners. | All threads hold same explicit state; stop request → join → GPU/event readers drain → finalizer/detach/free. Use existing thread and event primitives, not a new lifecycle service. |
| **Critical** | `rad0017/.../bc_ar_lifecycle_impl.h::build` publishes `rv.scratch/flags/seq`, then synchronizes peers; `acquire` holds failed published instances through deferred owner cleanup. | Freeing a failed rank's buffer before *other* ranks finish reading it may corrupt GPU state or hang host. | Inject failures immediately before and after publication, barrier timeout and peer-enable failure; count frees per device, verify GPU peers joined before release; defer cleanup via existing owner path. |
| **High** | `bc_ar_rendezvous_impl.h::rv_for(group)` resets slot payload when `owner` changes, after a group lookup. Numeric `slot_base_` can be reused by `rad_engine_release` (see startup.cpp). | Reset could overwrite live peer state during a failed or racing close/open. | Prove no later owner can claim group until old registry `fini`, rank threads, streams and `bc_shared_slot_retire_owner` complete. If this ordering is not guaranteed, fail on overlap rather than zeroing old state. |
| **High** | `moe_mailbox_ops.hip::mailbox_init` currently returns a raw `R3MoeMailboxDevice*` to `RadArgs::instance` without `fini`; the device includes owner/ticket/stamp pointers. | A new Engine can allocate/free memory while old copied kernel instance still retains addresses. | Small instance owns a borrowed generation reference; give all mailbox rows `fini`; ensure tapes and GPU work drain before the last release. Do not use a free-standing lease manager. |
| **High** | `register_mailbox` early `fstat`, `mmap` and seal/watch failures may leave fd, mapped pages, or other state acquired; current `HookState g` was never reset per Engine. | Sequential Engine starts can inherit resource leaks and stale helper/session bits. | One RAII release tail, tested after **each** failpoint: fd, mmap, register, watcher, helper-identity wait, policy, ledger, rank/device allocation and thread spawn. |
| **High** | The current closure proof checks per-rank sealed events and helper drain, but `Engine::drain_streams` handles additional compute/transfer/KV paths. | Draining the Ledger alone is not enough to free all device allocations. | Explicitly link existing `Engine::drain_streams`, rank joins and registry `fini_instances` before freeing transport, including owner upload streams/events. |
| **Medium** | `rad0017` already preserves timeout poison/arrival rollback; one-shot IPC has separate cleanup. | Duplicating the rendezvous in a generic remote-MoE context increases patch surface and can break mixed-DSO grouping. | Keep `rv_for(tp_group)` and shared-slot retire; only edit proven defects. Verify same-group cross-DSO/different-group and zero-width rank order. |

## Pass C — simplicity and reduction (mandatory architectural gate)

Current owned source sizes (measured with `wc -l`, revision `22388b657`): `rad0002/patch.py` **1,205**, `moe_remote_hook.cpp` **1,336**, `moe_remote_hook.h` **173**, `moe_mailbox_plugin.cpp` **45**, `moe_mailbox_ops.hip` **1,331**, `rad0017/bc_ar_lifecycle_impl.h` **478**. These are **not** a promise of deletion volume or a count of active patch anchors. Recompute exact old/new totals and also compare the generated payload, not only the Python patch definition.

| Candidate addition | Existing generic to reuse | Mandatory deletion/refactor | Verdict |
| --- | --- | --- | --- |
| A standalone `RemoteExecutionStateRegistry` or Engine factory manager | `Engine` owns members, `Plugin::handle`, existing `rad_engine_claim` | Must remove more than one old registry and lifecycle owner; unnecessary here | **Reject** |
| `BcMoeRemoteGeneration`/one binding | `Engine` member lifetime and RAII | Delete `HookLifetime`, `HookState g`, `g_plugin/g_mu/g_layer_maps/g_burst_active`, stateless callback getter copies | **Accept, minimal** |
| New `AdmissionQueue`, second ledger, `EpochService` | `admission_ledger.h`, `engine_admission_gate.h`, `engine_rank_coordinator.h` | None available: would duplicate live authorities | **Reject** |
| General-purpose plugin registrar / DSO loader | `Registry::plugin(name)->handle`, `RadKernelInfo::init/fini` | Existing loader already does this | **Reject** |
| Generic per-generation map framework | Existing TP-group and `rad_bc_shared_slot`; minimal read-only one-binding index if needed | Old `g_layer_maps` and rank-only selection; no new mutable execution registry | **Prefer no new framework** |
| New generic peer cleanup framework | `bc_ar_lifecycle_impl.h::free_rank_buffers`, shared-slot deferred cleanup, `Registry::fini_instances` | Delete duplicated cleanup branches only after real failpoint test | **Reuse; do not expand scope** |
| Global `RemotePolicy` singleton and getter compatibility APIs | Typed snapshot and shared policy predicate | Repeated `remote_policy()` lookups, `wanted/shadow/mask_only` implicit dispatch, old unused map replay | **Delete** |
| DSO initializer with "if enabled" | Static kernel tables; Engine-owned explicit bind | `HookLifetime`, live-count row gating, ctor open/close | **Delete** |
| New generic `StateMachine` class | Existing Ledger state plus one Engine binding state | No other user identified | **Reject; use small explicit enum/flags** |

**Quantified exit gate:** production-source inserted lines **less** than production-source deleted lines for the coherent full change, test/docs excluded. Include the source-level vs generated-payload distinction, `rad0002` anchor counts/affected upstream files, number of mutable singleton owners and duplicated callbacks. If a positive net diff is unavoidable for memory safety, include its exact reason, alternatives, cost and separate owner approval. Do not manufacture net savings by deleting safety checks, supported modes, IPC ordering, logging, tests, or error explanations.

**Maintenance gate:** the new code must have fewer independent answers to each question: (1) Who owns this Engine? (2) What is the helper policy? (3) Which parent is in flight? (4) Who can free this GPU buffer? Answer from Engine binding, one immutable policy snapshot, one Ledger ticket, and existing instance/group cleanup respectively. There must be **no separate adapter that copies these same answers into a second authority**.

## Simplification rejected approaches

1. A single global `HookState*` switched on each Engine start is a renamed singleton and fails live overlap/tape reuse.
2. TLS current Engine plus static callback dispatch introduces invisible state selection; worker, GPU and background threads do not necessarily share TLS lifetimes.
3. `std::map<int rank, State*>` picks a rank within multiple Engines and conflates Engine identity.
4. Embedding a new generation token in `StepTicket` changes a stable 16-byte device wire and creates a second identity; the parent ticket already has epoch/slot, while Engine identity is host-side.
5. Removing `rtld` boundaries by making `libr3moe` statically linked contradicts existing modular plugin loader and code-object selection.
6. Resetting shared TP state when owner integer changes without proving peer retirement is a UAF hazard, not a simplification.
7. Replacing `rad0017` with new group barriers/allocators duplicates a known implementation and threatens AQL wake/fences.
8. Keeping old C API under a compatibility flag doubles the paths to maintain and permits silent wrong-Engine execution.

## Resolution tracker (design review, not implementation evidence)

| ID | Needed before accept | Design disposition | Verification now |
| --- | --- | --- | --- |
| DR1 | Loader enumeration without helper | Static immutable rows, init is guarded | Not run |
| DR2 | One bound owner across all APIs | Context-aware single table + explicit direct operations | Not run |
| DR3 | Clean vs fault closure | One drain path + shared release tail | Not run |
| DR4 | Policy snapshot and HyperConn parity | Explicit snapshot across declare/runtime | Not run |
| DR5 | Group reuse and borrowed device lifetime | Existing kernel fini, owner/group and peer join | Not run |
| DR6 | Partial init and helper death | Existing RAII/deferred cleanup, injected failures | Not run |
| DR7 | No duplicate backend, measurable code reduction | Old globals/dispatch deleted, one minimal binding | Not run |
| DR8 | Mixed TP hardware and numerical canary | Four fresh leased runs + candidate canary | Not run |

**Conclusion:** design is **conditionally sound as a direction**, **not yet validated**. The decisive risk is whether the binding can be threaded through all declare/step callers without adding another service. S0 red tests and S1 diff must demonstrate that it can. If not, redesign toward *fewer* upstream touchpoints rather than keeping the old globals as a fallback. Preserve every original RUM06 clause and existing RV4303 findings.
