---
id: RCS14
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-13T03:53:18.506339+00:00'
breadth: ''
skill: advanced
created-by: bigcherry-1
priority: P0
work: L
---

# C05: remote-MoE topology from validated runtime counts (shared-memory protocol v26)

## Description

Umbrella: RCS12 (audit C05, VTP34 P0). The remote-MoE shared-memory protocol compiles in the topology: `kMaxEngineRanks=2`, `kMaxHelpers=4`, `kMaxSlots=4` (`moe-remote/moe_remote_shm.h:34-45`), `slots_fit()` refuses helpers*ranks > 4 (so even 1 helper x 3 ranks fails). Replace with validated per-segment counts (`Topology{helpers, ranks}`; slots = helpers*ranks, checked) in a versioned ABI migration (v25 -> v26), no compatibility reader. Dense TP3 (#863) does not exercise remote MoE and is not proof of this item.

## Steps

Gate A (card-free, first): freeze the v25 baseline outcomes for (1,1),(2,1),(1,2),(2,2); write failing tests for (1,3) and (2,3); TP1 combine non-alias, validator corruption, ACK identity; guard rejects the retired constants.
Gate B (v26 pure geometry): topology arithmetic with overflow/refusal predicates; page-aligned fixed header prefix (magic, version, header_bytes, segment_bytes, LayoutShape/shape hash, topology counts, offsets) with all count-sized tables (lane metadata, per-slot descriptors, per-helper PID/start/service, per-rank combine signals/chunks, request ring, ACKs) outside it; one result plane per used slot plus a dedicated combine plane per rank (the current unused-other-rank-plane trick aliases at TP1); `init` and `valid` as a producer/consumer pair; mutation of every offset/size/count/version/hash must be rejected.
Gate C (CPU): coordinator, worker_epoch, ACK/pidfd/ownership/forced-move, rank completion, worker threads/queues and the budget planner on the validated topology; ACK namespace helper [0,helpers), rank [helpers,helpers+ranks), service after the real ACK count; PREPARED/QUIESCING/COMMIT/RETIRING with lost/stale/mismatched ACK; a changed count recreates the segment.
Gate D (GPU, one change): `moe_mailbox_ops.hip` and `moe_remote_gpu.h` descriptor/stride/scratch together with host reference (`worker_hostref.h:163,261,363`), helper compare, worker combine, per-rank ticket/seal; replace `__shared__ step_nr/step_seq/from/row_of[...]` and `traced/seen_at` by descriptor-sized scratch (dynamic LDS only if it fits with the histogram, else preallocated device scratch); rework `mailbox_add_finish_kernel` (assumes kMaxSlots+1 threads in one block) keeping consumed_count visible before the seal; request first cache line identity and system-release/acquire ordering preserved.
Gate E (cutover): serving, v2 producer/measurement and `engine_adapters/radiance_layout.py:93-108` (parses constants from the C++ header today) move in lockstep; one native schema/validator; guard allowlist updated in the same PR.
Gate F (hardware): parity on a Flash-Next helper layout first (flashnext-2xtx TP1 two helpers; flashnext-tp2xtx-r9700 TP2 one helper), then TP3 remote helpers only on a layout with engine GPUs plus separate helper capacity.

## Detailed Solution & Technical Design

Real representation bounds stay and are checked with typed reasons, not as capacity literals: nonzero counts; uint32 participant masks (no 1u<<32; PlacementControl/Journal masks, `moe_remote_shm.h:130-156,192-219`); owner id vs the 0xff local sentinel (uint8 ownership); `worker_combine.h:75-81,199-215` uint8 row mask and `worker_share.h:15-20,56-85` eight-helper cap (specify a consistent mask or an explicit validated combine capability before allowing more than 8 helpers); lengths and HIP launch limits from types and device capabilities. Lane classes and per-lane ring depth are protocol parameters, not topology bounds. `shape_hash()` is model-only: use topology fields plus the manifest digest to refuse foreign attachments. The helper-count bootstrap word is published last: no attacher may validate half-built dynamic descriptors. Budget (`worker_footprint`, native P1 measurement, `Runner`) derives lane streams, queues and buffers from the same topology; remove the implicit 8-helper cap.

## Code Samples & Guidance



## Files

engines/radiance/components/moe-remote/{moe_remote_shm.h,moe_mailbox_ops.hip,moe_remote_gpu.h,moe_remote_hook.cpp,r3_moe_worker.cpp,worker_*.h,worker_hostref.h,worker_epoch*,worker_footprint*,component.toml,CMakeLists.txt}, engines/radiance/components/serving/{serving.cpp,v2/*}, tools/bigcherry/engine_adapters/radiance_layout.py, engines/radiance/model_constant_guard.toml.

## Validation

Card-free (no lease): named component CTests r3_moe_shm_test, r3_moe_epoch_test, r3_tp_exactly_once_test, r3_owner_generation_test, r3_admission_ledger_test, r3_mailbox_completion_test, r3_combine_f16_test, r3_combine_pool_test, r3_combine_epoch_test, r3_forced_move_test, r3_worker_footprint_test, r3_segment_version_test (take the build dir from the real component build); topology matrix (1,1),(2,1),(1,2),(2,2),(1,3),(2,3),(3,2),(4,1) plus zero/overflow/bad role/count-changing attach/corrupt offsets failing closed; every span in [0,segment_bytes), aligned, non-overlapping, identical on engine and helper; run the matrix again for the two synthetic LayoutShapes in RCS12; exactly-once consumer; TP1 combine plane never aliases; mutation checks (revert to kMaxSlots, old scratch offset, old ACK base, old result-plane index must break a test); `tools/tests/campaign/test_radiance_layout.py`.
Hardware (short leases, sequential, `launch-flashnext-best.sh start/stop`): baseline vs candidate on flashnext-2xtx then flashnext-tp2xtx-r9700; `helper-kld.json` and `helper-ab.json`, decode and prefill reported separately, deterministic output identity, route/combine counters, ACK/digest evidence. STOP on timeout, missing ACK, wrong digest, non-identical deterministic result or unexplained KLD. Performance acceptance is parity, not a gain.

## Effort & Risk

L, highest risk in the programme: a partial migration gives silent out-of-bounds DMA rather than a clean refusal. Needs a HIP design review of the finish kernel and scratch on the real device LDS limits before hardware. Each extra (rank, helper) slot multiplies lane streams, ring requests, mapped memory and scratch: refusal on budget is legitimate.

## Standards

No hand-picked constants; no compatibility layer (version gate rejects old segments, code and fixtures move together); fail closed; one coordinated change per gate; engine/plugin/helper protocol changes need both-sides layout tests.

## Acceptance Criteria

1. No `kMaxEngineRanks`, `kMaxHelpers`, `kMaxSlots`, `kResultPlanes`, `kLanes` or `kRingSlots` topology bound remains; every surviving limit is a named representation/kernel/hardware check with a typed refusal. 2. (1,3) and (2,3) validate card-free; v25 segments are refused; no shim. 3. Flash-Next helper layouts identical in output and speed/KLD parity. 4. RCS12 allowance for topology bounds expired or removed.

## Notes

Stage 2 of the radiance protocol programme (RCS12 staging). RHP02 slice B also bumps `moe_remote_shm.h`: land RCS14 first so the heat transport is added to v26, not a second bump. Depends on RCS13 only in that both touch `moe_remote_gpu.h`/`moe_mailbox_ops.hip`: land RCS13 first to avoid conflicts.

## Change Log

- 2026-11-13T03:53:18.506339+00:00 (created-by): Created by bigcherry-1
- 2026-11-13T03:54:03.474191+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:standards, section:acceptance_criteria, section:notes
