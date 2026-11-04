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



PROGRESS 2026-11-03 slice C step 2 (branch feat/rhp11-slice-c-admission): per-step admission ledger with derived capacity (route banks + 1); per-rank device-memory consumption counts sealed to the host once per parent behind a radiance event; fail closed on an invalid seal; no-op placement epoch cycles through PlacementControl; helper drain via the burst word (lanes keep main's serving/waiting loops); drain-time lane identity published by the helper CPU at drain ACK. Hardware: identical text throughout; cycles complete (27-30 per cycling arm) with no hang. Performance history: the first correct form cost +2.6% 1-stream decode; rounds 1-6 reached +0.8-1.0%; hybrid builds (swapping only the helper binary) put the cost in the helper; a cumulative helper bisect (main engine; H0 main sources, H1 shm layout, H2 epoch coordinator, H3 lane loops, H4 completion stores: +0.14/+0.06/(stall)/+0.09/+0.62%) and bisect 3 (round 6 without the stores: +0.81% to +0.30%) isolated the two per-request system-scope stores of done_epoch/done_slot in lane_publish_done. Round 7 (71c6af599) restores main's completion kernel and moves those words to the drain ACK: 1-stream decode +0.14% (cycles off) / +0.28% (cycles on) vs main, within noise (runs/rhp11-c2-ab7). The intermittent +60% after-idle stall seen during this work also occurs on main, so it is not slice C's. Follow-ups: RHP15 (media encoder passes vs admission), RHP16 (calibrated move triggers).

PROGRESS 2026-11-03 slice D (PR #557, branch feat/rhp11-slice-d-engine-gpu): two owner-bitmap generations per engine rank, selected by the pass StepTicket; route-mark identity check once per request; exact per-parent consumption seal and r3_moe_all_ranks_consumed; full transaction driven by one production EngineRankCoordinator (engine RESERVED, per-rank upload/readiness, quiesce, commit, retire, bounded REOPENED); helper 0 seals the filtered bootstrap owner map once and engine rank setup waits for it; precommit faults abort forward with serving untouched; local-only admission when the patch policy declares no mailbox ops. Hardware path to parity: first form +1.5% 1-stream (per-route owner lookup) and every cycle aborted (no engine RESERVED, then an unsealed bootstrap digest -> INVALID_MAP, then the engine refusing the unsealed map at kernel declaration); four fix rounds plus two independent reviews. Final evidence on radiance 1.16 (runs/rhp11-d-ab5, ab6): decode step 1-stream -0.42% (off) / +0.08% (cycling), 6-stream -0.31% / -0.29%, prefill unchanged, text identical, 27-54 same-map epoch cycles per cycling arm with no fault; 6-stream MTP acceptance varies 0.56-0.64 per round in every arm and run (noise, not slice D). The handoff-disabled configuration cannot start Flash-Next on this layout (engine OOM without helpers), so B1's fix is covered card-free through the real patch policy. Next: slice E (helper GPU slot migration, worktree feat/rhp11-slice-e-helper-slots, in progress).

PROGRESS 2026-11-04 slice E (PR #558, branch feat/rhp11-slice-e-helper-slots): helper expert arena of fixed slots with the layout budget kept as an expert-weight budget (a part-2 attempt to count lane buffers inside it would have cost 573 experts on flashnext-4gpu and was reverted); extra buffers checked against hipMemGetInfo; Request::generation_slot moved to +36 so bank selection reads no new cache line; live helper-to-helper moves (local/helper split immutable), committed-budget CAPACITY refusal, per-expert private-stream uploads, READY after last event and bank, reclamation after REOPENED, CLEANED-gated proposals with deadlines. Two independent reviews (4 + 3 findings) fixed. Hardware on radiance 1.16 (runs/rhp11-e-ab1..ab4): decode within noise of main at 1 and 6 streams, prefill unchanged, text identical, 28-29 same-map cycles per cycling arm with no fault. No real move has run on a GPU yet: slice G (forced-move mode) is in progress on feat/rhp11-slice-g-forced-move, stacked on E.

## Change Log

- 2026-11-01T22:51:14.829553+00:00 (updated-by): Updated: section:title, order=10, work='L', priority='P0', section:description, section:steps, section:detailed_solution, section:validation, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20261101_225732_planning-a-staged-roadmap-for_4092
- 2026-11-01T22:57:50.943098+00:00 (updated-by): Updated: section:ledger-events
- chg_20261101_233905_design-spec-for-moving-experts_5737
- 2026-11-01T23:39:09.725180+00:00 (updated-by): Updated: section:ledger-events
- chg_20261102_002949_groundwork-for-moving-experts_1337
- 2026-11-02T00:29:56.603122+00:00 (updated-by): Updated: section:ledger-events
- chg_20261102_022722_groundwork-for-live-expert-pla_5619
- 2026-11-02T02:27:26.869346+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-02T02:40:47.403962+00:00 (updated-by): Updated: section:notes
- chg_20261102_024204_planning-records-updated-with_6210
- 2026-11-02T02:42:16.533832+00:00 (updated-by): Updated: section:ledger-events
- chg_20261102_055505_groundwork-for-live-expert-pla_6885
- 2026-11-02T05:55:09.623001+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-03T00:02:36.089233+00:00 (updated-by): Updated: section:notes
- chg_20261103_000523_the-engine-can-now-pause-drai_4810
- 2026-11-03T00:05:27.792971+00:00 (updated-by): Updated: section:ledger-events
- chg_20261103_190740_the-engine-side-of-live-expert_4409
- 2026-11-03T19:07:44.686043+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-03T19:08:17.653290+00:00 (updated-by): Updated: section:notes
- chg_20261104_005244_helpers-can-now-take-over-and_6656
- 2026-11-04T00:52:48.511888+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-04T00:52:52.689176+00:00 (updated-by): Updated: section:notes
- chg_20261104_065043_experts-can-now-be-moved-betwe_3090
- 2026-11-04T06:50:47.247787+00:00 (updated-by): Updated: section:ledger-events
