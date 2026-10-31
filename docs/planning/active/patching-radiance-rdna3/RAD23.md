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
3. Combine methods behind one interface: off, cpu (AVX2/F16C, already on feat/combine-fp16), gpu (CPU futex readiness, then a GPU reduction on a device-local stream with explicit system-visible publish, presence masks and per-row fallback). 1 helper resolves to off.
4. auto: measured once per host and cached - P2P availability, per-card host-link bandwidth, CPU features - choosing method and combiner helper (widest link; tree reduce when N-1 planes over one link loses). Per-model profile may override.
5. Harness first: replay/combine bench for each method with 1, 2 and synthetic N planes before end-to-end.
6. KLD gate and 4-pair A/B per method on the build server vs the parent build.

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

No fixed helper count in moe-remote code or protocol; off/cpu/gpu/auto selectable and auto derived from measured topology; the chosen method on the build server beats combine-off on prefill with tg not worse and KLD within the gate.

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


Handoff 2026-10-23 (historical): the feat/combine-fp16 branch subsequently merged as PR #404 (cpu method with F16C, measured +5-8% prefill). The
GPT addendum on the N-helper protocol (req_bigcherry_3bb6b763ab97) FAILED to run - resubmit it to dev-gpt-agent
(new session) before redesigning the layout. #404 (fp16 combine, F16C cpu method) is merged; start a new slice. Owner priority question open: do RAD20 (no polling, frees two cores) before the gpu method?

2026-10-27, DeepEP/MORI-EP brief: GPU combine variants to measure against the CPU F16C method (0.31-0.33 ms a large step, +5-8% prefill): F1 GPU reads both mapped helper rows; F2 helper 0 keeps its reduced row in VRAM and adds helper 1's mapped row (avoids round-tripping its own result; most promising); F3 F2 + staged DMA output (see the result-output item). Helper 0 launches only after the existing futex handoff from helper 1: no pending GPU wait. Same FP32 accumulation order and FP16 overflow fallback; KLD gate. CPU stays selectable; auto from measured topology.

## Execution gate review (2026-10-23)

**Split RAD23 into sequential implementation PRs; do not ship a monolithic topology change.** Before editing the ABI, retrieve and reconcile the recorded N-helper design addendum `req_bigcherry_3bb6b763ab97`; this planning note does **not** assert that private review was retrieved here.

1. **RAD23-A: ABI/layout only.** Separate runtime helper count H from engine rank count R (two axes). Define alignment-checked, overflow-safe offsets/lengths with `layout_for(H,R,classes,shape)`, process-shared header validation, generation and ownership invariants, protocol version negotiation and deterministic refusal of mismatched versions; keep legacy behavior for current H=2/R=1. Test H=1/2/3 plus R=1/2 on CPU fixtures, boundary allocations, malformed headers, offset overlaps and zero-length optional regions. No performance claim or GPU method in this PR.
2. **RAD23-B: method negotiation.** Normalize `off|cpu|gpu|auto` in one protocol field with explicit requested/selected method, capability checks and engine acknowledgment. H=1 always off. Explicit unsupported method fails startup with an actionable reason; auto may choose off, never silently substitute on explicit selection. Preserve AVX2/F16C semantics and row-level FP16 overflow/fallback.
3. **RAD23-C: GPU combine.** Use **CPU futex handoff** for cross-process readiness; do not wait on unproven host-mapped GPU signal words. Validate helper-local VRAM residency, uploaded presence masks, stream ordering, system-visible publication and stale generation rejection. Require exactly-once sum of present helper rows and equivalent overflow markers; no combine consumer may read partially written output. Compare off/cpu/gpu KLD and reproducible per-step latency with N=1/2/3 synthetic planes before hardware A/B.
4. **RAD23-D: auto topology policy.** Use measured critical-path cost including host-link read/write bandwidth, GPU DMA, row count, helper count, CPU load and method startup cost; cache by topology/PCI identifiers plus ROCm version with invalidation and audit log. Never assume P2P; never assume a wider link is always the fastest combiner. Benchmark real current hardware, then gate defaults on measured wins; retain explicit methods for unknown topologies.

**Dependencies:** RAD18 evidence first; RAD20 CPU-pressure reduction before GPU-combine comparisons; RAD19 model-derived expert metadata contract enters RAD23-A, not a hard-coded default. RAD21 begins only after H/R layout/generation stability. Keep RAD22 after RAD21 simulation. A successful H=2 result does not prove N-helper correctness.

## Change Log

- 2026-10-23T10:48:18.492687+00:00 (created-by): Created by agent
- 2026-10-23T10:48:36.867246+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
- 2026-10-27T02:35:27.322130+00:00 (updated-by): Updated: section:notes

## Plan 2026-10-31 (first consumer: the RX 6900 XT as a third helper, RCS07)

Read of moe_remote_shm.h (551 lines) and the 129 kMaxRanks sites. In the shipped configuration (a one-rank engine, two helpers) the
per-"rank" slots are really per-helper slots: Header says "helper h owns bitmap h and serves rank slot h". So three things share one
constant: engine ranks R (TP), helpers H, and the slot index the two are aliased onto. Memory is the constraint that makes a bigger
constant wrong: a result plane is max_rows x hidden x 2 bytes (about 20 MiB at 4096 x 2560) and the layout holds R x H of them in
pinned host memory, so a compile-time 4 x 4 would cost 320 MiB against 80 MiB today on a host that runs at 74 GiB pinned already.

1. Layout from counts. `LayoutShape` gains `ranks` (R) and `helpers` (H); `layout_for` sizes owner bitsets [H], route and activation
   slots [S] with S = the slot count the producer side uses (max(R, H) while helpers alias onto slots; see step 4), result planes
   [R][H]. `valid` rebuilds every offset from the header's own R, H (it already rebuilds from the shape; today it also pins
   `rank_count == kMaxRanks`). kMaxRanks stays only as a protocol bound (kMaxEngineRanks = 2, kMaxHelpers = 8) with static_asserts,
   like kMaxLayers.
2. Header arrays that are per helper (`helper_pid`, `helper_start`, `CombineSignal::lane/seq`) move to [kMaxHelpers]; the 24-word
   reserved block is recomputed; kVersion 18 -> 19. `LaneSignal` lanes stay per (class, slot).
3. Ownership. `select_owned` alternates two helpers (`who ^= 1u`, `held[who]`): generalise to H helpers by assigning each expert to
   the helper with the lowest (held bytes / budget) that can take it, heaviest first, ties by helper index. With H = 2 this must give
   the same ownership as today (golden test over the Flash-Next heat file), so existing runs do not move.
4. Engine side (`moe_remote_hook.cpp`, `moe_mailbox_ops.hip`, patch rad0002): `RankState rank[kMaxRanks]` and `published[]`, `from[]`
   become runtime-sized from the header; routes go to the slot of the helper that owns the expert (already the rule for H = 2).
5. Combine stays a two-helper feature: `R3_MOE_WORKER_COMBINE` requires H == 2 and says so by name; with H > 2 the engine reads the
   H planes and adds them (as it does with combine off). The N-plane combine, the method interface and `auto` (RAD23 steps 3-4) are a
   separate slice.
6. Heterogeneous helpers: the backend (moe_backend.h `backend_for`) is already per architecture; each helper loads its own plugin
   (libr3 for gfx1100, a gfx1030 build for RDNA2). Launcher: one worker per card, `R3_MOE_WORKER_COUNT=H`, per-card budget from
   `R3_MOE_WORKER_MIB`.

Tests first (no hardware): moe_remote_shm_test with (R, H) in {(1,1), (1,2), (1,3), (2,2)}: layout sizes, `valid` accepting each and
refusing a segment whose header disagrees, ownership golden for H = 2 and balance for H = 3; a CPU model of the engine's slot choice.
Then hardware: helper-ab at H = 2 against the parent build (must not move), then H = 3 with the 6900 XT.

GPT design review: the earlier requests (req_bigcherry_97cebda33722, req_bigcherry_3bb6b763ab97) are recorded in Steps; a new review
of this plan could not be submitted from this session (no ag-* gateway tools after the restart on 2026-10-30).

