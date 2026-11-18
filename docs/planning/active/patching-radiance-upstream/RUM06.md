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

## Detailed Solution & Technical Design

Actual extension callbacks include admit, bind, cancel, before_pass, host_done, rank_done, poll and shutdown. Preserve their admission ledger rules; do not replace them with a new task scheduler. Parent identity, placement epoch and helper session identity must remain distinct. A TLS ticket can carry per-rank current work, but it cannot select process-global policy for a different engine or survive a reused worker thread without clearing.

Upstream g_model is now indexed by rad_build_slot(ctx)/rad_slot(c); collective rendezvous uses a tp_group key. Engine rank 0 is not globally unique. Bind a narrow local EngineExecutionState reference to the context at creation; resolve that reference outside the MoE inner loop. Keep data structures allocation-free for normal route/combine calls.

Upstream AQL and kernel changes adjust system acquire, scalar-cache invalidation, doorbell flush and wake stores. Preserve the target's documented chain_fenced contract when reusing two-shot, duplex and once paths (rad0021/rad0024/rad0032/rad0033). Source-level fence removal is not an acceptable cleanup. Reuse the shared collective group with one scratch owner and release it after all users drain.

Failure handling is part of a working transition: helper death must name a failed helper/session, not let an absent remote result be read or misreported as generic OOM. Existing numerical-quality policy remains in force; do not add replication or a new fallback path to dodge missing owners.

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

## Effort & Risk

L: the main hazards are reused ranks/groups, stale callbacks and unfinished reservations during free. Reuse proven ledgers and make teardown order explicit; don't redesign transport algorithms.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

Callbacks/maps/policy belong to the admitted engine and group; all reservations reach an exact terminal state and maps outlive their consumers. Target chain/fence semantics and rank ownership survive. Unsupported multi-model helper operation refuses cleanly, and the selected production layout passes its minimal canary.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM02, RUM03, RUM05. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisites RUM02/RUM03/RUM05. Coordinate RCS14 topology without absorbing unrelated backend/kernel redesign. Reuse the current protocol epoch rather than introducing another generation format.

## Change Log

- 2026-11-18T01:28:27.514165+00:00 (created-by): Created by codex
- 2026-11-18T01:39:45.710628+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:43:17.981906+00:00 (updated-by): Updated: section:effort_risk
- 2026-11-18T01:56:58.746185+00:00 (updated-by): Updated: section:code_samples
