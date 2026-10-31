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

## GPT review of the plan, 2026-10-31 (req_main_a51e4069b049) and what changes

Verdict: approve the R/H separation with corrections; approve the fdot2 kernel conditionally; prove a third helper helps end to end
before investing in a new gfx1030 GEMM.

Accepted and now part of the plan:
- A synchronous MoE layer waits for its slowest helper (T_layer ~ max(T0,T1,T2) + T_combine), so a slow helper can lengthen the
  layer. Assignment must be latency-aware, with a work budget for the gfx1030 helper and a legal zero-work mode; cold experts can
  still receive large batches when selected.
- A cold expert that lives only on the slow helper cannot be dropped when the helper is over budget: the first experiments use
  replicated experts (a second copy on an XTX) so the budget can shed work; migration is a later item.
- Classify each of the 129 kMaxRanks sites by semantic axis (engine rank, helper index, result producer, lane index, storage
  capacity) before changing it; do not replace them mechanically. Named hazards: moe_lane_host.h StoredWeight arrays and
  worker_lanes.h lane statistics are engine-rank axes; worker_harness.h addresses results as rank * kMaxRanks + r (take an explicit
  stride from the validated layout).
- Slot lifetime: activation staging shared by H helpers needs a generation and a participant mask (or count); an old completion must
  not release a recycled slot. Keep result-completion as the release point unless an earlier activation_consumed ack measures as a gain.
- Preserve the H = 2 path exactly at first (ownership, route offsets, dispatched rows, results, speed), not just the ownership map.
- valid() validates R, H, S bounds, alignment, overflow, monotonic non-overlapping offsets and the mapped length before it derives
  any pointer from header fields.
- Tests to add: (R,H) = (2,3) and (2,8), maximum sizes, invalid and half-initialised headers, overflow, helper crash, late
  completion, ring wraparound, repeated start/stop, and a deliberately slow helper that finishes several requests late.
- Quality: helpers finishing in different orders change fp16 rounding, so the gate is KLD, not per-layer relative error.

Order of work (replaces the earlier list):
1. Instrument the H = 2 run: T_engine_publish, T_helper_start/done per helper, T_engine_all_done, T_combine, bytes and expert rows per
   helper. This is the baseline critical path.
2. Layout and protocol split, H = 2 behaviour unchanged.
3. H = 3 with the third helper owning nothing (startup, shutdown, idle, no regression).
4. H = 3 with a bounded gfx1030 budget (0, 5, 10, 20 percent of routed work, replicated cold experts), existing kernels, 128/512/2048-token
   decode and 1024/4096-token prefill, identical routing. Useful only if total time falls consistently with no decode regression.
5. Only then the native fdot2 grouped GEMM (RCS07) - justified if gfx1030 is on the critical path and 18.3 ms -> 4.1 ms would change it.
6. dot4 requantisation, compact result planes, adaptive placement only where profiling justifies them.

Not accepted as stated: the review calls the E4M3-to-f16 expression (code & 0x7f) << 7 | (code & 0x80) << 8 invalid. It is
r3_e4m3x4_to_f16 in libr3/r3_compat.h, used by the fp8 GEMMs that pass rad-kbench (2083 checks): E4M3 (bias 7) and half (bias 15) share
the subnormal rule, so every finite code maps to value x 2^-8 exactly; only the E4M3 NaN encoding differs (it maps to a finite
half), and the kernel's callers do not feed NaN codes. The PCIe figure is the same order as the review's own per-layer arithmetic
(about 4.5 ms of ideal link time a layer for 2048 rows, times the layer count).

## Baseline: helper timing at H = 2, 2026-10-31 (RAD23 step 1)

Flash-Next, v1.14.0 lab build, R9700 engine + two XTX helpers, `launch-flashnext-best.sh` at the new defaults (1M context, 2 sequences),
`R3_MOE_WORKER_TRACE=10` (every 10th finished request; stamps from worker_harness.h: engine clock for publish / wait entered, helper
clock for the helper's own span). Random-word prompts tokenise to about 5.8 tokens a word, so the prompts were 16.8K and 66.5K tokens.
Run output: `/work/development/tmp/bigcherry/rad23/trace.out`, helper logs `/work/bigcherry/flashnext/helper*-1031-020842.log`.

| Lane | Rows a request (median) | Helper GPU span (median / p90) | Requests traced |
|---|---|---|---|
| Small (decode), helper 0 / helper 1 | 4 | 127 / 187 us and 126 / 189 us | 1,034 / 1,033 |
| Large (prefill), helper 0 / helper 1 | 2,048 | 3,482 / 3,860 us and 3,672 / 4,197 us | 228 / 228 |

Prefill ran at 3,263 tok/s (16.8K) and 3,017 tok/s (66.5K), decode 94 tok/s for 512 tokens. A 2,048-token chunk takes about 630 ms, so
a 3.5 ms helper span in each of the 48 routed layers is about 170 ms: the two XTX helpers are busy roughly a quarter of a prefill
chunk and a fifth of a decode step. The engine, not the helpers, is the critical path in this layout. That is room for a slower
third helper, if its share is small: a gfx1030 helper measured 4.5x slower on the grouped GEMM would take about 16 ms a layer for an
XTX-sized share, which is over the roughly 13 ms a layer a chunk allows, but about 4 ms for a quarter of that share. Caveat: this is
the helpers' own time, not a measured critical path; the engine-side wait stamps still need reading per layer before the budget is set.

## Site classification by GPT, 2026-10-31 (req_rad23-n-helpers_abc3ee290b59, read-only on this worktree)

Axes: R engine rank, H helper, S request slot (max(R, H) while helpers alias onto slots), P result plane, L lane arithmetic, C capacity,
A the alias helper == request slot == apparent rank. Findings that change the plan:
- The dangerous sites are the A sites: worker_shape.h:74 (`lane % kMaxRanks == helper_index`), moe_remote_hook.cpp:441-443 (published
  flags), moe_mailbox_ops.hip:141,144,272,308,409,567 (slot and generation indexing), moe_remote_gpu.h:254,387,429,467 (request
  validation and lane decode), moe_remote_shm.h:135-136,159-160,202-203,218,298,332,365.
- Result planes are today the diagonal [slot][slot] (`result_offset[slot * kMaxRanks + slot]`, moe_mailbox_ops.hip:460,528 and
  worker_harness.h:328). [producer rank][helper] is NOT the same convention: the migration must translate explicitly, and every reader
  and writer (host layout, the device kernels, the helper) takes the index from one shared helper, `result_index(producer, helper)`.
- GPU-visible: Header, Request and CombineSignal change their member offsets, so the engine and the helper must share the version bump;
  moe_mailbox_ops.hip step_nr / step_seq / published / traced / seen_at / from capacities and the published[S+1] launch dimensions
  change with them. Changing only the host layout builder is not enough.
- R stays a compile-time bound (RankState, StoredWeight arrays in moe_lane_host.h / moe_lane_common.h, worker_lanes stats).
- H = 3 with kMaxRanks left at 2 would overflow helper_pid[2], helper_start[2], combine lane[2] / seq[2] and the device arrays
  traced[2], seen_at[2], from[2], step_nr[2], step_seq[2]; `%2` lane decode would map slot 2 onto slot 0; moe_mailbox_ops.hip:318,336
  stop at two helpers.
- Shared definitions to add: one `LayoutShape{R, H, S}`, `lane_of(slot, cls)`, `result_index(producer, helper)`,
  `published_generation_index(S)`; combine remains an explicit H = 2 special case.

## Stage 1 done 2026-10-31: kMaxRanks split into three named axes (no layout change)

`kMaxRanks` is gone. `kMaxEngineRanks`, `kMaxHelpers` and `kMaxSlots` (all 2) replace it, each of the 129 uses taking the axis GPT's
classification gave it; `kResultPlanes` and `result_index(slot, out_rank)` replace the open-coded `slot * kMaxRanks + r`, and
`lane_of` is `cls * kMaxSlots + slot`. Proof that nothing moved: header and request sizes, header field offsets, the legacy layout's
byte count and plane offsets and the lane numbering are identical between origin/main and this branch (compared by a program built
against both headers) and are now asserted in `moe_remote_shm_test.cpp`; the whole component builds (worker, libr3moe, tests);
`--compare` on an XTX gives the same rel_l2 as before (8 rows 6.7e-4 / 6.2e-4, 64 rows 1.75e-3 / 1.15e-3). Next: stage 2 raises the
helper and slot bounds and makes the layout runtime-sized from (R, H, S).

## Stage 2 done 2026-10-31: segments laid out for their slot count, up to four helpers, weighted shares

Protocol 19. `Header::slot_count` is set at init (`slots_for(helpers)`: never fewer than the 2 engine ranks); route and activation
staging, owner bitsets and result planes exist only for the slots in use, `layout_for`/`init` take the slot count, and `valid` also
refuses a header whose unused slots are not all zero or whose slot count disagrees with its layout. `kMaxHelpers` and `kMaxSlots` are 4;
engine ranks stay 2; the combine remains a two-helper feature (`kCombineHelpers`, off with more helpers). Cost per extra slot:
74,227,712 bytes (70.8 MiB: route + activation + two 20 MiB result planes) - three helpers 212.6 MiB of pinned host memory against 141.9
for two; the lane ring grew by 8 KiB for the four-slot bound (lane numbering is `class * 4 + slot`). `R3_MOE_WORKER_SHARES="w0,w1,.."`
weights the helpers' parts of the expert ranking (default equal = the round robin it always was); a part of 0 owns nothing.

Evidence: `r3_moe_shm_test` (segments for 1-4 slots, tampered headers, unused-slot offsets, short mappings), `r3_worker_share_test`,
`r3_moe_backend_test` pass; the component builds for gfx1030 / gfx1100 / gfx1201. Hardware at H = 2 against the baseline above, same
prompts: decode 512 tokens 5.436 s (5.440), prefill 16.8K 4.89 / 5.06 s (5.14 / 5.41), 66.5K 21.78 s (22.05); helper GPU span medians
127.4 us (127.3) on the small lane and 3,489 us (3,482) / 3,674 us (3,672) on the large lane. Not yet: H = 3 on hardware (needs the
6900 XT helper to start with an empty or small share - see the share-0 path, untested with a GPU), the engine-side per-layer wait
stamps, and the budgeted experiments (step 4).

