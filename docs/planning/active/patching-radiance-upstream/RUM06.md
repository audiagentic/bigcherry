---
id: RUM06
order: 6
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:27.514165+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P0
work: L
---

# Port engine-scoped remote MoE and mixed-TP collective lifetimes

## Description

Port remote callbacks, parent tickets, layer maps and cached policy to engine/TP-group lifetime; reconcile per-group rendezvous, timeout rollback and chained fences. Preserve mixed TP ownership and all issue-ahead/cancel/drain rules; initially refuse unsupported multi-model helper operation.

## Steps

- Audit rad0002's g_plugin, layer maps, static RemotePolicy values and g_rank_ticket TLS; replace process-global execution state with a state object owned by the admitted engine generation.
- Bind the state once at engine/rank creation and pass typed step/ticket context to before_pass and route/combine hooks. Keep existing plugin extern-C struct-size/version checks; migrate consumers together if that ABI changes.
- Port rad0017 shared rendezvous onto upstream rv_for(tp_group), preserving timeout arrival rollback. Share the intended group between libr3/libr4d without making different engines meet by rank number alone.
- Port RadBuildCtx ownership/pipelining fields and update ABI layout rows; keep rank ordering, zero-width/tail slices, DFlash geometry and complete tensor ownership.
- Reuse admission_ledger, engine_admission_gate, engine_rank_coordinator and mailbox completion primitives. Test exactly-once reserve/bind/cancel/host_done/rank_done semantics for issue-ahead and varying verify row count.
- Define shutdown order explicitly: close admissions -> cancel unused reservations -> drain admitted ranks/helper/GPU work -> unregister callbacks -> retire maps/scratch -> free engine. Handle partial initialization using the same cleanup path.
- For the initial cutover require one remote-MoE engine and refuse unqualified multi-instance use; don't set rad_arch_multi_instance merely to silence the loader.

- Add the lifecycle bridge points omitted from the original file list: engines/radiance/components/moe-remote/moe_remote_hook.cpp and direct r3_moe_mailbox_device / r3_moe_owner_generation_{upload,ready} entry points. Replace DSO-static owner registration with explicit bind/open and stop/detach using an opaque engine-generation context; capture immutable policy at bind.
- Make normal stop close admission, cancel only an unbound tail reservation, allow bound parents to finish/seal/host-complete, drain rank work, then detach callbacks and free maps/scratch. Keep fault interruption as a distinct wake path; do not abort the rank barrier before clean joins.
- Fix ABI43 libr4d ArInst partial-initialization cleanup: each allocation must be owned as soon as acquired, and all published peer pointers must remain valid until group participants unwind. Reuse the normal per-rank release logic on failure.
- Preserve collective declaration order for zero-width and uneven ranks and retain exactly-once bridge callback behavior.

## Detailed Solution & Technical Design

Actual extension callbacks include admit, bind, cancel, before_pass, host_done, rank_done, poll and shutdown. Preserve their admission ledger rules; do not replace them with a new task scheduler. Parent identity, placement epoch and helper session identity must remain distinct. A TLS ticket can carry per-rank current work, but it cannot select process-global policy for a different engine or survive a reused worker thread without clearing.

Upstream g_model is now indexed by rad_build_slot(ctx)/rad_slot(c); collective rendezvous uses a tp_group key. Engine rank 0 is not globally unique. Bind a narrow local EngineExecutionState reference to the context at creation; resolve that reference outside the MoE inner loop. Keep data structures allocation-free for normal route/combine calls.

Upstream AQL and kernel changes adjust system acquire, scalar-cache invalidation, doorbell flush and wake stores. Preserve the target's documented chain_fenced contract when reusing two-shot, duplex and once paths (rad0021/rad0024/rad0032/rad0033). Source-level fence removal is not an acceptable cleanup. Reuse the shared collective group with one scratch owner and release it after all users drain.

Failure handling is part of a working transition: helper death must name a failed helper/session, not let an absent remote result be read or misreported as generic OOM. Existing numerical-quality policy remains in force; do not add replication or a new fallback path to dodge missing owners.

ABI43 review findings: `moe_remote_hook.cpp` has process-global `HookState` and DSO-static registration. `RTLD_LOCAL` does not create an owner per Engine. The remote device and owner-generation APIs bypass the `BcMoeRemotePlugin` callback surface and must use the same bound owner. Clean shutdown cannot call the existing interrupt callback: ledger polling refuses after interruption, and `Engine::serve` currently aborts the barrier before joining. Clean stop must drain bound work; fault stop must interrupt to wake blocked peers. In candidate `libr4d/r4d_ar_entry.hip`, `ArInst::build` holds allocations in locals until late publication; failure can delete the instance without normal release and leak host/VRAM allocations. Deallocation must wait until no peer can dereference the shared address.

## Code Samples & Guidance

Proposed state ownership sketch:
```text
Engine admitted generation
  owns RemoteExecutionState
    owns helper session + immutable PlacementView + admission ledger
    references upstream TP group and bound kernel handles
Rank step
  carries parent ticket + generation; never re-resolves config
Shutdown
  admissions closed -> admitted work drained -> callbacks unregistered -> storage freed
```

Target source landmarks: upstream libr4d/r4d_ar_entry.hip rv_for(group), rv_barrier(rv,world); local BcMoeRemotePlugin callbacks in rad0002; existing admission_ledger.h and engine_rank_coordinator.h. These are extension points to reconcile, not a request for another callback framework.

### Deep scan implementation refinement

Trace bc_moe_remote_register/g_plugin -> admit/bind -> rank TLS -> before_pass -> route/combine -> host_done/rank_done -> shutdown. Reuse admission_ledger/engine_admission_gate/engine_rank_coordinator, bind engine/generation/group once and remove function-static policy caches. Reuse mailbox epoch; don't introduce a second ticket or epoch service. Preserve target fence/acquire/wake ordering.

Source evidence and removal conditions: docs/reviews/radiance-upstream-2026-11-18/CONSOLIDATION.md. Use patch-footprint.json to inspect every overlapping patch before changing an attachment site.

## Files

- engines/radiance/patches/rad0002_moe_remote_handoff/patch.py
- engines/radiance/patches/rad0017_vtp14_n3_shared_rendezvous/patch.py
- engines/radiance/patches/rad0018_vtp14_n2_tp_ownership/patch.py
- engines/radiance/patches/rad0019_vtp14_n2_tp_ownership_fp8/patch.py
- engines/radiance/patches/rad0020_vtp14_tp3_pipelined_prefill/patch.py
- engines/radiance/patches/rad0025_multiseq_pipelined_prefill/patch.py
- engines/radiance/components/moe-remote/admission_ledger.h
- engines/radiance/components/moe-remote/engine_admission_gate.h
- engines/radiance/components/moe-remote/engine_rank_coordinator.h

## Validation

Extend existing ledger/coordinator fixtures for cancel-unused, partial-init failure, reused rank/group, drain-before-free and helper death. Build touched plugins once. One representative mixed-TP helper-on canary belongs to RUM09; no all-layout sweep or prolonged stress unless a specific lifecycle failure demands it.

Focused gates: engine-generation bind/unbind across repeated Engine lifetimes; clean stop with a bound issue-ahead parent; fault wake/timeout and partial-registration failure; allocation failure after each ArInst allocation with zero leaked host/VRAM bytes; cross-DSO same-group rendezvous plus different-group isolation; zero-width/uneven rank declaration order; exactly-once bridge call counts. Reuse admission_ledger, rank coordinator, and shared-slot fixtures. The wider hardware route and mixed-TP cutover remain RUM09.

## Effort & Risk

L: the main hazards are reused ranks/groups, stale callbacks and unfinished reservations during free. Reuse proven ledgers and make teardown order explicit; don't redesign transport algorithms.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

Callbacks/maps/policy belong to the admitted engine and group; all reservations reach an exact terminal state and maps outlive their consumers. Target chain/fence semantics and rank ownership survive. Unsupported multi-model helper operation refuses cleanly, and the selected production layout passes its minimal canary.

Callbacks, direct device operations, owner upload/ready and collective state resolve through exactly one live engine/TP generation. A clean stop closes admission and drains bound work before resource teardown. Faults unblock peers without stale owner/address reuse. Every partial init path frees rank-local resources only after peer unwind. No process-global helper policy or DSO constructor registration remains in the production path.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM02, RUM03, RUM05. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisites RUM02/RUM03/RUM05. Coordinate RCS14 topology without absorbing unrelated backend/kernel redesign. Reuse the current protocol epoch rather than introducing another generation format.

Astra source review ASTRA-CONSOLIDATION.md: initial scope is the admitted single-engine generation, not a general multi-engine factory. Preserve independent helper model-byte ownership verification. Reuse existing parent tickets and mailbox epochs. One existing partial-start/drain fixture plus route/combine activation in RUM09 is sufficient unless a real failure requires expansion.

2026-11-19 Sol implementation refinement: start with the pre-publication ArInst allocation leak in ABI43 composed `libr4d/r4d_ar_entry.hip` (`build()` locals 424–470, `acquire()` failure 558–559). Make each acquired address rank-locally owned immediately and reuse the ordinary release tail only before publishing `rv.scratch/flags/seq`; do not free published pointers until peers unwind. A host-only fake-HIP test can call the real exported init path and inject failure at each allocation, asserting exact-once matching frees and no peer publication; verify the libr3 rewrite preserves the fix. Separately inspect `r4d_ar_oneshot_2rank_exact.hip` allocation cleanup on memset/sync failure. Keep post-publication unwind, engine-owned callback context, clean-stop drain, and helper lifecycle as separate required RUM06 work; this small slice does not complete the item.

Checkpoint cbdd21f5b frees rank-local collective buffers on pre-publication initialization failure, including GPU memset and stream-sync refusal. The exact 2.2 host fault fixture passed 23 cases; ordered 33-patch apply and libr3 rewrite verification passed. The existing historical N3 tests stop at rad0046's required-state gate. This checkpoint leaves coordinated post-publication unwind, engine-scoped remote callbacks and clean-stop drain open.

Checkpoint c5abd3b47 poisons and wakes failed rendezvous and holds published owner buffers until joined rank workers and device contexts have drained. Four rendezvous cases, exact ordered 33-patch replay, libr3 rewrite and host/HIP syntax checks passed. Live mixed-TP, clean stop and remote callback ownership remain open.

Checkpoint bedf51edb consolidates rank attachment rollback, clears the result pointer on refusal and destroys completion events on their owning device during shutdown. Exact ABI44/ROCm syntax and four focused host fixtures passed. This preserves the frozen canary SDK; engine-scoped remote lifecycle, clean-stop drain and hardware proof remain open.

2026-11-20 owner re-scope (RUM09): the bump must carry every config main serves, so this item is now a bump blocker: the Qwen 27B TP3 configs (gfx1100, gfx1100, gfx1201 engine ranks) and the Flash-Next tp2xtx configs must start and qualify on 2.2. The TP patches rad0016-rad0033 are already in serving-core on the candidate; what remains is the mixed-rank code-object, ordinal-reversal, group-rendezvous and span tests and a TP3 27B start/canary/paired comparison on the 2.2 build.

## Change Log

- 2026-11-18T01:28:27.514165+00:00 (created-by): Created by codex
- 2026-11-18T01:39:45.710628+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:43:17.981906+00:00 (updated-by): Updated: section:effort_risk
- 2026-11-18T01:56:58.746185+00:00 (updated-by): Updated: section:code_samples
- 2026-11-18T02:19:52.290526+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:23:21.830497+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:26:23.178450+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4303
- 2026-11-19T02:21:54.505617+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:validation, section:acceptance_criteria
- 2026-11-19T09:12:22.140456+00:00 (updated-by): Updated: section:notes
- 2026-11-19T23:03:33.981692+00:00 (updated-by): Updated: section:notes
- 2026-11-19T23:20:26.906081+00:00 (updated-by): Updated: section:notes
- 2026-11-20T01:23:16.000046+00:00 (updated-by): Updated: section:notes
- 2026-11-20T08:50:11.218145+00:00 (updated-by): Updated: section:notes
