# RUM06 refactor economics — proposed code reduction and generic reuse

**Applies to:** [engine-generation design](ENGINE-GENERATION-DESIGN.md), [RUM06 implementation draft](IMPLEMENTATION-PLAN-DRAFT.md), [adversarial review](DESIGN-REVIEW.md). This is a required **implementation/design gate**, not proof that source has already changed.

## Additional source finding: RUM02 already owns patch consolidation

The existing [RUM02 plan](../../planning/active/patching-radiance-upstream/RUM02.md) documents a successful consolidation pattern: rad0017's typed rendezvous/peer lifecycle now lives in `rad0017/src/*.h`, while rad0009 moved its native manifest source into `components/serving/`. The result was fewer upstream anchors and easier source review, not merely smaller Python files. The same pattern is relevant to rad0002.

Observed in the task worktree:

- `rad0002_moe_remote_handoff/patch.py` has **1,205 physical lines** and embeds its five production C/C++ sources as raw strings: `abi/bc_moe_remote.h`, `core/place/bc_moe_remote.cpp`, `core/place/bc_moe_remote_engine.h`, `core/place/bc_moe_remote_place.h`, and `arch/common/bc_moe_remote_block.h`. This makes production C++ hard to compile, navigate, test and refactor in isolation.
- rad0016 and rad0017 already hash-bind their owned `src/` files to the containing `patch.py`; rad0051 also uses hash-bound owned sources. The patch identity source hashes must change along with source bytes; do not silently make a pinned patch mutable.
- `tools/tests/patch/test_rum06_registry_lifetime.py` currently reads raw C++ via AST and asserts the **old globals** exist. Those tests need to become negative tests for the deleted registry and positive tests for bound-engine semantics; do not preserve a superseded global merely to pass them.
- RUM02's latest tested patch consolidation measured upstream-owned anchors and total generated payload, and verified exact source + rewritten libr3 bodies. Follow its reporting format and tooling, **do not build a second patch-analysis framework**.

## Simplification requirement A — source ownership extraction (part of S1, not a separate new feature)

Move `rad0002`'s large BigCherry-owned C++ literals into a named `rad0002/src/` source directory **only when the migration refactors the same bodies**, or reuse an already existing canonical shared component where appropriate. Keep `patch.py` declarative: minimal typed `FilePatch` attachments, explicit SHA-256 digests of the owned source inputs, and existing patch framework guards. Match rad0017's proven `src/` + hash-binding convention and RUM02's patch-source review procedures.

- No duplicated copy of the C++ payload in both `patch.py` and `src/`. The generator must read the one source of truth.
- Do **not** create a new patch source reader in a standalone RUM06 utility. A generic pinned-input helper belongs in the *shared patcher* only if the existing rad0016/rad0017/rad0051 call sites are migrated to it together under RUM02's authority and the resulting helper removes duplicated code. Otherwise keep the already documented per-patch digest check.
- Do not move the C++ to a new engine-root directory if it is only consumed by this patch; follow the existing engine-scoped patch owner. If the C++ can be compiled as an existing serving/component source (as rad0009 did), reuse that component instead of duplicating a generated copy.
- Verify `FilePatch(create=True)` rejection on collisions, all anchors and generated output, and compile the extracted C++ against the composed exact Radiance 2.2 ABI.
- Do not count moved lines as deletions: report **semantic production change** (old emitted C++ vs new emitted C++), **authoring source footprint** (`patch.py` vs src), and **upstream patch anchors** independently.

This extraction is justified by deleting giant embedded string declarations and reducing patch-definition complexity. Do not expand upstream hooks merely to perform the move.

## Simplification requirement B — explicit deletions in the same slice

| Delete or consolidate | Replace with / reuse | Proof that refactor actually occurred |
| --- | --- | --- |
| `g_plugin`, `g_mu`, per-call locked callback getter/copy boilerplate | One instance-taking callback table owned by `Engine` | Search confirms old symbols removed from *emitted* C++, and callbacks never select a process-global engine |
| `g_layer_maps` and pre-registration replay | Per-generation map stored once during declaration | A→B model test sees no stale map; source lacks replay branch |
| `RemotePolicy remote_policy()`, repeated `bc_serving_remote_settings()`, implicit `wanted/shadow/mask_only` globals | One frozen Engine policy and one shared handoff predicate | Placement, MoE block, HyperConn and runtime tests all use same policy; no global alternate selector |
| `g_burst_active` global and independently inferred policy | Existing Engine admission burst transition bound to generation | Exactly one start/idle transition per Engine, no second dispatch mutex |
| `HookState g` and `HookLifetime` static ctor/dtor | One opaque `HookState` under Engine RAII binding | No DSO registration/open/unload side effects, repeat starts work |
| Rank-only `r3_moe_mailbox_device`, `owner_generation_*` and static context in `LiveRankOperations` | Explicit bound instance; `RadKernelInfo::init/fini` and existing epoch coordinator | Wrong-group/stale-engine call fails without HIP operation |
| Distinct clean/fault teardown code branches freeing the same allocations | One rank-resource release tail, with distinct clean vs fault *entry* behavior | Failpoint tests prove every address freed exactly once and never before peer retirement |
| Old test contracts asserting the wrong global registry behavior | Red/green lifecycle tests running the resulting extracted C++ | Tests pass by actual API behavior, not string-presence compatibility |

**Scope guard:** `moe_mailbox_ops.hip` contains substantial kernel logic; do not rewrite kernels or change fences to make the line count look better. Deletion should come from removing obsolete *host-side* authority, callbacks, loaders and duplicate cleanup, not transport math.

## Simplification requirement C — measured outcome / no false savings

At commit `22388b657`, baseline physical line counts were `rad0002/patch.py=1205`, `moe_remote_hook.cpp=1336`, `moe_remote_hook.h=173`, `moe_mailbox_plugin.cpp=45`, `moe_mailbox_ops.hip=1331`, and `rad0017/src/bc_ar_lifecycle_impl.h=478`. These are starting observations, not an instruction to delete a fixed line count.

Final review must provide all **five** numbers, before and after:

1. **Total emitted C/C++ production LOC** across touched managed extensions and generated patch files (source extraction is LOC-neutral unless logic is actually deleted).
2. **Total maintained authoring LOC**, accounting for any new `src/` files and helper.
3. **Non-create and upstream-owned patch anchors**, number of upstream target files, and composed patch application identity using the RUM02 footprint report procedure.
4. **Mutable state/lifecycle authorities:** old global callback registry, duplicate policy reader, DSO lifetime, cross-DSO duplicate registry. Final target is *zero* of these, except the single non-owning helper exclusivity gate.
5. **Distinct callable paths and symbols:** old stateless C ABI exported functions removed, new context-aware ones exactly one each; no duplicate fallback, shim, or special-case per-backend context lookup.

**Required outcome:** emitted production LOC net **negative** and lower control-path complexity, or a signed owner-approved exception based on a specifically measured memory-safety invariant. Source extraction alone may have a large `patch.py` deletion count but **does not** satisfy the net-reduction gate; nor does removing safety checks, diagnostics, hardware qualification or tests.

The reusable pieces are already present: `Engine` RAII ownership, `Registry::plugin(handle)`, generic `RadKernelInfo::init/fini`, `rad_build_slot`, `rad_slot`, `rad_bc_shared_slot`, `rv_for(tp_group)`, `admission_ledger`, `RankCompletion`, `EngineRankCoordinator`, and `bc_shared_slot_retire_owner`. A new generic abstraction with only one user **fails** the simplification test unless it replaces at least two proven duplicated bodies and demonstrably reduces maintenance. Avoid scope-creeping into all kernel/collective backends.

## Proposed per-commit deletion ledger

| Slice | Old code removed in same slice | Reused generic or owned seam | Acceptance |
| --- | --- | --- | --- |
| S0 | No production changes | Existing host fixtures | Red tests + verified source/patch footprint |
| S1 | Global policy/registry/map getters/replay, embedded-literal authoring (if extraction) | Existing Engine + plugin registry + typed policy | First binding; helper-off and second-owner refusal; measured delta |
| S2 | DSO HookLifetime/global HookState and rank-only APIs | Existing `Ledger`, `EngineRankCoordinator`, kernel init/fini | Repeated generations + direct API + rollback tests, measured delta |
| S3 | Split interrupt/normal close path and duplicated resource cleanup | Existing Engine drain, rank deferred cleanup and shared slot | Clean/fault/peer unwind tests, measured delta |
| S4 | Any residual obsolete path and redundant upstream anchor | RUM02 patch footprint and source hashing pattern | Composed source byte/ABI checks and net negative delta |
| S5 | No code changes unless hardware exposes a real fault | Existing GPU test harness and leases | Four fresh named hardware receipts |

This is an implementation **design gate** and must not be entered in Planning/Ledger as a completion record. Preserve RUM06 `in_progress` until code, tests, hardware and review close the actual acceptance clauses.
