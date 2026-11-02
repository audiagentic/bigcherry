---
id: RHP11
order: 10
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-31T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Stage 1: placement-epoch mailbox protocol v23 (live helper ownership changes)

## Description

Stage 1 prerequisite of the live-placement roadmap (README). Owner direction 2026-11-01: expert placement across the helpers changes live, never by restart. The v22 mailbox (engines/radiance/components/moe-remote/moe_remote_shm.h) carries per-helper ownership bitsets that the engine uploads to its GPU once at start (moe_remote_hook.cpp) and that worker_heat.h's proposals never change. v23 adds a placement epoch: a PlacementControl block (active_epoch, pending_epoch, transaction_state, owner_map_digest, ready_mask, fault_code, two generation slots each holding the complete canonical owner map, one byte an expert, sized from the model shape), and Request gains placement_epoch + generation_slot written before the sequence. A completion is valid only when sequence and epoch both match. Breaking revision: every engine and helper caller migrates together, no v22 reader or writer remains.

## Steps

1. Spec the v23 layout (PlacementControl, Request fields, layout_for, size/offset asserts in C++ and HIP).
2. Engine side (moe_remote_hook.cpp, moe_mailbox_ops.hip): per-step epoch snapshot; route against that generation's owner table; device copies of both generations.
3. Helper side (worker_lanes.h, worker_combine.h, r3_moe_worker.cpp): epoch capture per request, pre-queued lane handling, combine epoch scoping, result-consumed ack.
4. Coordinator (one, in the engine hook or a helper-0 role - decide in the spec) running the state machine with timeouts and fault codes.
5. Card-free tests of the protocol (moe_remote_shm_test.cpp and a new epoch test).

## Detailed Solution & Technical Design

Epoch state machine (one transaction at a time): ACTIVE(N) -> PROPOSED -> RESERVING -> COPYING -> PREPARED(N+1) (all uploads done, pointer tables ready, digests and ACKs from every engine rank, every helper and every residency manager) -> QUIESCING (hold new-step admission; let issued steps finish under N, including pre-queued lane waits, MTP draft/verify, CPU combine jobs and the engine's consumption of combined rows) -> COMMIT(N+1) (publish active epoch with the existing system-scope fences; all participants ACK) -> RETIRING(N) -> ACTIVE(N+1). One epoch per whole model step (all 48 MoE layers and MTP). Immutable double-buffered owner maps and helper pointer tables; the inactive buffer is prepared while N serves and never recycled until old GPU references retire. Combine jobs carry their epoch and participant set and never span a flip; combine.done is not a lifetime fence for the output plane - add an engine result-consumed acknowledgement. A small transaction journal (last committed generation, candidate digest, participant state) for crash recovery. Rollback is a new forward epoch, never a reused number or a restored bitmap.

## Code Samples & Guidance



## Files



## Validation

Card-free: structure sizes/offsets/alignment; dynamic helper and rank counts; one owner per eligible expert; digest agreement; invalid epoch/digest; lost or delayed ACK; duplicate commit; aborted staging; epoch monotonicity; concurrent propose/commit; failed helper before and after commit. With RHP06: forced-churn stress (>=1,000 transitions) and the stage 1 end-to-end gate (RHP12).

## Effort & Risk

Depends on: RHP06. Added by the 2026-10-31 GPT review of this plan.

## Standards



## Acceptance Criteria

v23 is the only protocol; every transition passes the card-free suite; with RHP06 a live expert move under serving traffic completes with exact per-step ownership agreement and no missing, duplicate or stale-epoch expert work.

## Notes

Plan source: GPT staged plan 2026-11-01 (gateway session ses_335f497feb31449a, request req_main_8f893e8abe79), reviewed against the code. Delivered together with RHP06 as one stage 1 milestone; neither is complete alone. Deterministic mode (--deterministic) keeps placement fixed.

PROGRESS 2026-11-02: spec merged (#508, docs/design/moe-remote-placement-epochs-v23.md, offsets compile-checked). Slice A merged (#513): v23 layout - canonical owner generations, PlacementControl, journal, ACKs, per-request epoch/slot, consumed stamps; old ownership-bitmap API removed; hardware identity vs main passed (24,571 owned experts, byte-identical single-stream text, step time within noise). Slice B (#515): CPU epoch controller (worker_epoch.h), helper slot ledger (worker_residency.h), peak-load heat policy with bounded swaps (worker_heat.h), card-free tests incl. exhaustive policy oracle. Next: slice C (engine step admission in the managed rad0002 patch) - research on step boundaries, MTP lifetime, graph replay and the last consumer of the result plane in progress (gateway request req_main_e70e447f5747).

## Change Log

- 2026-11-01T22:51:14.829553+00:00 (updated-by): Updated: section:title, order=10, work='L', priority='P0', section:description, section:steps, section:detailed_solution, section:validation, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20261101_225732_planning-a-staged-roadmap-for_4092
- 2026-11-01T22:57:50.943098+00:00 (updated-by): Updated: section:ledger-events
- chg_20261101_233905_design-spec-for-moving-experts_5737
- 2026-11-01T23:39:09.725180+00:00 (updated-by): Updated: section:ledger-events
- chg_20261102_002949_groundwork-for-moving-experts_1337
- 2026-11-02T00:29:56.603122+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-02T02:40:47.403962+00:00 (updated-by): Updated: section:notes
- chg_20261102_024204_planning-records-updated-with_6210
- 2026-11-02T02:42:16.533832+00:00 (updated-by): Updated: section:ledger-events
