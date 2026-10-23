---
id: RAD23
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-23T10:48:18.492687+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# moe-remote: helper count and topology as runtime values (no fixed 2), selectable combine methods

## Description

Owner rules (2026-10-23, CLAUDE.local.md): code for unknown topologies - no fixed helper/device counts; topology-dependent methods stay selectable with a derived auto. The moe-remote protocol and helper hard-code two: kMaxRanks = 2 (moe_remote_shm.h) indexes result planes [rank][helper], combine.lane/seq are 2-element arrays, and R3_MOE_WORKER_COMBINE requires helper_count == 2. Make helper count (1..N) and engine ranks runtime values with separate axes, and make the helper-row combine a selectable method: R3_MOE_WORKER_COMBINE=off|cpu|gpu|auto (later p2p).

## Steps

1. GPT design review (req_bigcherry_97cebda33722 + addendum req_bigcherry_3bb6b763ab97): protocol for N helpers, GPU-through-host combine race-freedom, overflow fallback, auto selection, prior art.
2. Protocol: helper count and engine ranks as runtime header fields; per-helper arrays sized from them; layout_for derives region sizes; version bump. Remove every hard-coded 2 / kMaxRanks used for helpers.
3. Combine methods behind one interface: off, cpu (AVX2/F16C, already on feat/combine-fp16), gpu (combiner helper's stream waits device-side on the other helpers' done words, reads their rows from host memory, adds its own in VRAM, writes one fp16 row with the per-row fallback marker). 1 helper resolves to off.
4. auto: measured once per host and cached - P2P availability, per-card host-link bandwidth, CPU features - choosing method and combiner helper (widest link; tree reduce when N-1 planes over one link loses). Per-model profile may override.
5. Harness first: replay/combine bench for each method with 1, 2 and synthetic N planes before end-to-end.
6. KLD gate and 4-pair A/B per method on Brutus vs the parent build.

## Detailed Solution & Technical Design

Inventory 2026-10-23 (grep over engines/radiance/components/moe-remote, 176 hits in 15 files: moe_mailbox_ops.hip 41,
moe_remote_shm.h 40, r3_moe_worker.cpp 18, moe_remote_gpu.h 16, worker_combine.h 14, moe_remote_hook.cpp 12, others <= 7).
Root cause: kMaxRanks = 2 (moe_remote_shm.h:21) is both the engine-rank (TP) axis and the helper bound.
- Layout: lanes (kLanes = kMaxRanks * kClasses, lane_of/lane_class), route/activation slots [kMaxRanks], owner bitsets
  [kMaxRanks helpers], result planes [kMaxRanks * kMaxRanks] (moe_remote_shm.h:74-75,129-135,153-155,176-198,269-345);
  the header has a runtime helper_count but the layout ignores it; validation requires rank_count == kMaxRanks (:325).
- Combine: two helpers only - Header::combine.lane/seq[kMaxRanks], helper1_done, lane1/seq1, {q0, q1}, planes
  [0][0] + [1][1] -> [0][1] (worker_combine.h:95-141); engine uses published[kMaxRanks] as the combine slot.
- Engine hook/kernels: RankState rank[kMaxRanks], helpers <= kMaxRanks refusals, owner copies sized kMaxRanks
  (moe_remote_hook.cpp:55,160,284,323,341-349); 41 sites in moe_mailbox_ops.hip (step_slot, published[], from[kMaxRanks]).
Design direction: separate axes - engine ranks R and helpers H as header fields; every per-rank / per-helper region and
array sized from them in layout_for; result planes [H][R]; combine over H planes into one fp16 plane per R.



## Code Samples & Guidance



## Files



## Validation

Component tests with helper counts 1, 2 and 3 (shm layout, combine correctness per method, fallback rows); KLD gate passes for cpu and gpu; helper-ab A/B per method; grep shows no hard-coded helper count.

## Effort & Risk



## Standards



## Acceptance Criteria

No fixed helper count in moe-remote code or protocol; off/cpu/gpu/auto selectable and auto derived from measured topology; the chosen method on Brutus beats combine-off on prefill with tg not worse and KLD within the gate.

## Notes


GPT design review req_bigcherry_97cebda33722 (2026-10-23), adopted order:
1. Finish the AVX2/F16C CPU A/B (actual combine cost).
2. GPU method = GPU arithmetic with the existing CPU futex handoff (sleeping wait, no polling): helper 0 keeps its
   local result in a device buffer (today lane_reduce_tokens_v writes straight to host-mapped memory,
   moe_remote_gpu.h:480-491,761-775), waits on helper 1's done futex, uploads a per-token presence mask, then one
   block a token: bf16->fp32 add of present rows, fp16 RNE, overflow -> marker, and helper 0's bf16 row written only
   for overflow rows (sparse fallback); completion published after system-visible writes.
3. Device-side cross-process waits (hipStreamWaitValue32 on host-registered shared memory) are NOT used: ROCm
   documents the API for hipMallocSignalMemory only, and a stranded device wait would deadlock shutdown
   (hipStreamSynchronize at r3_moe_worker.cpp:516). Revisit only if measurements justify it, with explicit
   generation-tagged success/abort signals.
4. Method negotiated in Header::combine (requested vs selected, logged once); the engine follows the selected method.
   Explicit unavailable method fails init; auto falls back. auto predicts critical-path latency per method from
   cached one-time microbenchmarks (mapped GPU read/write, GPU reduce, AVX2/F16C reduce, engine read) keyed by PCI IDs,
   ROCm version and topology; profile may override method / min rows.
5. Protocol v17. Consider a per-row fallback bitmap instead of the column-0 marker.
6. Remaining yield() polling in the large-lane loop (r3_moe_worker.cpp:499-503) belongs to RAD20.

Follows RAD18 (combine). fp16 combined row passed KLD (p99 +0.039). CPU scalar fp16 conversion erased the gain (~1 ms/step); F16C version under A/B. Owner: the combine should run on the GPUs where possible.

## Change Log

- 2026-10-23T10:48:18.492687+00:00 (created-by): Created by agent
- 2026-10-23T10:48:36.867246+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
