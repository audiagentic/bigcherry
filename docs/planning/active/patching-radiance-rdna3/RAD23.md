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

## Correction and the real workload for the third helper, 2026-10-31 (from the route-heat file)

Earlier notes in this item called the third helper's experts "cold" while the XTX helpers' were left unnamed; the launcher owns the
HOTTEST experts on the XTXs (`heat.rank0.hot-first`, ranking order = hottest first, each helper takes its share of the top). From that
file (`/work/bigcherry/runs/perf51/heat.rank0.hot-first`, 48 routed layers x 512 experts, 4,421,376 recorded routes; recorded
2026-10-17, the workload that profile run used):

- Two XTX helpers at 22 GiB each own 18,640 experts (9,320 each, 2.417 MiB an expert) = 99.3% of routes.
- The other 5,936 experts (14.0 GiB) carry 0.68% of routes (0.2-1.0% a layer; 54-206 experts a layer). They stay in the engine's pinned
  host pool today and are copied to the R9700 when a chunk uses them (the log's "prefill staging ... up to 3 layers ahead").
- In a 2,048-token chunk (20,480 routes) about 140 routes, roughly 3 a layer, land on them; in a decode step well under one a layer.
- A 6900 XT with a 16 GiB card holds all 5,936 (14.0 GiB). So its job is the tail of the ranking, not a share of the hot experts, and the
  slow emulated grouped GEMM (4.5x an XTX at 256 rows) runs at about 3 rows a layer: launch time, not matrix time. What it buys is not
  speed but the engine's host expert pool (about 14 GiB of pinned host RAM, and the staging copies).

What this changes: `R3_MOE_WORKER_SHARES` (weighted round robin) would hand the 6900 XT hot experts, which is the wrong placement. The
third helper needs a TAIL placement: helpers 0..H-2 take the ranking as now, up to their budgets, and the last helper takes the
experts that remain, with a budget of its own for each helper (R3_MOE_WORKER_MIB is one value for every process today, the launcher
sets it per process). This is RHP07 (heat-tiered placement) made concrete. Caveat: one profile of one workload; a coding or
long-document workload can make some of the 5,936 hot, and a route to the tail must then still be correct (it is: the helper serves
it) and only the speed changes.

## First run with the 6900 XT as the third helper, 2026-10-31

Flash-Next, R9700 engine, two XTX helpers (22 GiB each) and the RX 6900 XT as helper 2 on the existing kernels (a gfx1030 libr3 built
with R3_ONLY and the emulated 16x16x16 tile, RCS07), `R3_MOE_WORKER_PLACEMENT=tail`, `R3_MOE_WORKER_MIBS=22528,22528,<tail MiB>`. The
tail helper takes the COLDEST leftovers that fit its budget (from the end of the heat ranking backwards). Run through
`/work/development/tmp/bigcherry/rad23/sweep3.sh`: decode is the mean of two 384-token rates (416 minus 32 tokens), prefill is one request
each on three slices of the repository docs (natural text), one run a configuration, so decode is good to about +-5% (story 90.2 to 103.0 between runs).

| Tail budget (experts held) | Decode story / code / count, tok/s | Natural prefill 25K / 14K / 72K, tok/s | Pool slots freed on the engine |
|---|---|---|---|
| none (two helpers) | 90.2 / 142.8 / 156.2 | 3,297 / 3,053 / 2,998 | 0 |
| 2,048 MiB (847) | 103.0 / 139.8 / 154.7 | 3,094 / 2,819 / 2,941 | 687 (1.7 GiB) |
| 3,000 MiB (1,241) | 92.8 / 138.8 / 155.2 | 2,736 / 2,943 / 2,919 | 1,081 (2.6 GiB) |
| 6,144 MiB (2,542) | 96.8 / 131.2 / 153.8 | 2,084 / 2,845 / 2,804 | 2,307 (5.6 GiB) |
| 14,336 MiB (5,931, all the leftover; an earlier run, hottest-leftover order, prompts from a second script) | 92.8 / 129.0 / 143.6 | 1,282 / 2,168 / 2,051 | 5,465 (12.9 GiB) |

What the runs show:
- It works end to end on the engine: three helpers, a 5,931-expert tail on a gfx1030 card, no errors in any log.
- The heat file understates the tail's traffic on this model. With natural text the 14 GiB tail got 343 routes in a 2,048-token chunk
  (1.7%, against 0.68% in the profile); with random-word prompts it got 2,919 (14%), and 9 of the 38 routes of a 4-row decode step. The
  profile (2026-10-17) does not include layer 48 (the nextn / MTP layer, all zero, so ranked coldest and therefore in every tail),
  which every draft pass uses; to be checked.
- The cost is the 6900 XT's call, not its traffic: a large-lane call with 343 routes took 12.5 ms against 3.5 ms for an XTX (the emulated
  tile costs per expert tile, not per row), so with a big tail the card is the slowest in every layer and prefill falls 30-60%. A
  small tail keeps it off the critical path: up to about 3 GiB prefill loses 2-17% (the 17% is the first request after start, which
  also loads the card's code), decode is within the run noise.
- The gain is host memory and nothing else: 0.7 to 5.5 thousand pool slots (1.7 to 12.9 GiB of the engine's pinned host pool).
- Found and fixed on the way: the result plane a helper writes was `[slot][slot]`, which does not exist for slot 2 (offset 0, a write over
  the segment header). `out_rank_of(slot, world)` now gives the output rank (the slot when the engine has several ranks, 0 when one
  rank is served by several helpers); every reader and writer uses it. The launcher's stop only knew two helpers.

Next: the prefill cost is the emulated grouped GEMM; a GEMV-style path for the few rows an expert gets on a tail card (the helper's own
fdot2 kernel) or a native dot2 tile would remove it, and is what makes a large tail affordable. The run script is a scratch copy; the
launcher change (a third helper, per-helper budgets, tail placement) goes in after #465 so the two do not conflict.

## The few-rows kernel as the 6900 XT's large-lane kernel, 2026-10-31

No new kernel was needed: the helper already has `lane_moe_rows` (RAD06, `R3_MOE_WORKER_ROWS`), dot products with no WMMA, one block an
(expert, 32 outputs), the expert's weights read once for all its rows. gfx1030 now takes it by default (`R3_MOE_WORKER_ROWS=0` turns it
off), because the plugin's grouped GEMM there is the emulated tile, which costs per expert tile and not per row.

Same sweep as above (`sweep4.sh`; 6900 XT helper only, tail placement, natural-text prefill, one run a row):

| Tail budget | Prefill 25K / 14K / 72K, emulated tile | Prefill, few-rows kernel | Two helpers | Large-lane call on the 6900 XT, median (p90) |
|---|---|---|---|---|
| 6,144 MiB (2,542 experts) | 2,084 / 2,845 / 2,804 | 1,828 / 3,038 / 2,942 | 3,297 / 3,053 / 2,998 | 2.6 ms (4.1) with 46 routes; an XTX: 3.5 ms with ~7,700 |
| 14,336 MiB (5,931 experts) | 1,282 / 2,168 / 2,051 | 1,233 / 2,679 / 2,550 | same | 5.4 ms (19.7) with 253 routes; was 12.5 ms |

Decode with the kernel on: story / code / count 97.2 / 131.1 / 150.5 (14 GiB) and 95.9 / 136.5 / 153.6 (6 GiB), against 90.2 / 142.8 / 156.2
for two helpers - inside the +-5% run noise except code, which is 3-8% lower. The first request after a start is the slow one in
every configuration with the third helper (25K: 1,233 and 1,828), the card's code objects loading; steady state is the 14K and 72K figures.
The 6900 XT's call is faster than an XTX's only because it carries a handful of routes (46 against ~7,700); per route it is far slower.

Correctness evidence, and what is not shown:
- `--compare` against the host reference with the grouped GEMM is unchanged on both cards (rel_l2 1.75e-3 / 1.15e-3 at 64 rows).
- `--compare` with the few-rows kernel on gfx1030, 256 rows (the large lane, lanes 16 / 17; at 64 rows and below compare uses the small
  lanes and the GEMV, so only 256 rows reaches the kernel): rel_l2 1.24e-3 and 1.14e-3 against the host reference (the emulated
  tile: 1.15e-3 and 1.02e-3); rocprofv3 shows `lane_moe_rows` for both projections (9.4 ms and 3.5 ms for the two requests at 1,024
  routes, against 13.3 and 5.0 ms for the emulated tile). An earlier note here said the run hung: it was my 250 s timeout; a compare
  run takes 80 s (9 rows) to 240 s (256 rows), almost all CPU reference time, with 17-26 ms of GPU time in total.
- End to end at temperature 0, same prompts, `text.py`: the two-helper text and the three-helper texts (emulated tile, few-rows kernel)
  diverge from each other at the same place (116 characters of the story, 76 of the code prompt; the count prompt is identical), while the two
  three-helper runs agree with each other for 257 and 220 characters. So the early divergence comes from three helpers (the merge adds
  three planes in a different order), not from the kernel. Text agreement is not a numerical gate: Flash-Next KLD with three helpers is
  the gate and has not been run.
- Bug found by the compare run, fixed here: `worker_harness.h` looped over `kMaxSlots` (4 after stage 2) in a segment laid out for 2;
  `--compare` and `--replay` crashed. It now loops over the segment's `slot_count`.

Next: the KLD gate for the three-helper layout, then the launcher change.

## KLD gate for three helpers, 2026-10-31 (`rad23-kld-h3-10310518`, `rad23-kld-h3emu-10310527`)

Flash-Next, the existing quality harness (`ab-benchmark`, quality-only, 8,240 positions) against the 2026-10-23 exact reference, a
control arm of two helpers (`h2`) beside the arm under test; settings as in `kld-h3.json` (`R3_MOE_WORKER_DEVICES="0 1 3"`,
`R3_MOE_WORKER_MIBS=22528,22528,14336`, `R3_MOE_WORKER_PLACEMENT=tail`). The helpers are started by the harness; the logs confirm the 6900 XT
helper owned 5,931 experts and (arm 1) used the few-rows kernel. The harness thresholds are the repository's own.

| Arm | KLD mean | median | p99 | p99.9 | top-1 | PPL (ref 9.111) |
|---|---|---|---|---|---|---|
| two helpers (control; identical in both runs and to the v1.14.0 bump run) | 0.05518 | 0.01419 | 0.5557 | 1.6424 | 89.745% | 9.0651 |
| three helpers, 14 GiB tail, few-rows kernel | 0.06343 | 0.01613 | 0.6655 | 2.0756 | 88.883% | 9.2974 |
| three helpers, 14 GiB tail, emulated tile (`R3_MOE_WORKER_ROWS=0`) | 0.06150 | 0.01554 | 0.6548 | 2.0553 | 88.762% | 9.1151 |

Both three-helper arms pass the gate (limits: mean +0.02, median +0.01, p99 +0.2, p99.9 +0.5, top-1 -1.0 point, PPL ratio +0.05) and both
are clearly worse than the control: mean KLD +15% / +11%, p99.9 +0.43 / +0.41 (limit 0.5), top-1 -0.86 / -0.98 points (limit 1.0). It is not
the few-rows kernel (the emulated tile is nearly as far from the control), and per-layer `--compare` agreed with the XTX. Candidate
cause under test: the 5,931 tail experts were computed by the engine's own kernels from its host pool and now run through the helper's
path (E4M3 widened to f16), so the loss would scale with the number of experts moved. A run with 2 and 6 GiB tails checks that.

## KLD against tail size, and what the numbers do and do not say, 2026-10-31

More KLD runs (same harness, 8,240 positions, exact reference; `rad23-kld-h3sizes-10310535`, `rad23-kld-x2-*`):

| Layout (all from the same helper and plugin build) | Experts on helpers | KLD mean | median | p99 | p99.9 | top-1 | PPL | Gate against its own control |
|---|---|---|---|---|---|---|---|---|
| two XTX helpers (control) | 18,640 | 0.05518 | 0.01419 | 0.556 | 1.642 | 89.745% | 9.065 | - |
| + 6900 XT tail 2 GiB | +847 | 0.05555 | 0.01444 | 0.538 | 1.752 | 89.502% | 9.019 | pass |
| + 6900 XT tail 6 GiB | +2,542 | 0.06444 | 0.01539 | 0.706 | 2.414 | 89.320% | 9.056 | **fail** (p99.9 +0.77, limit +0.5) |
| + 6900 XT tail 14 GiB | +5,931 | 0.06343 | 0.01613 | 0.666 | 2.076 | 88.883% | 9.297 | pass (p99.9 +0.43, top-1 -0.86) |
| one XTX helper only (control 2) | 9,320 | 0.06213 | 0.01545 | 0.637 | 2.514 | 88.847% | 8.894 | - |
| + XTX tail 6 GiB (helper 1 on an XTX) | +2,542 | 0.06529 | 0.01590 | 0.793 | 2.360 | 88.859% | 8.893 | pass |

- A 2 GiB tail on the 6900 XT is indistinguishable from the control. 6 GiB fails the harness on p99.9, 14 GiB passes by 0.07.
- It is not proportional to the experts moved (6 GiB is worse than 14 GiB), and it is not the few-rows kernel (14 GiB with the
  emulated tile: mean 0.0615, top-1 88.76%).
- The control itself moves with the layout: one XTX helper (the engine computes the other 15,700 experts) is 0.0069 above two XTX
  helpers, with no 6900 XT anywhere, and a 6 GiB tail on an XTX moves that by +0.0032. Moving experts between the engine's own
  kernels and the helper path changes KLD by about +-0.005 to 0.01 whichever way, so the 6900 XT arms are inside that spread; they are not
  shown to be worse than an XTX tail, and not shown to be equal (the baselines differ, and p99.9 rests on about eight positions).
- Why the paths differ by that much is open: the helper widens E4M3 to f16 (R3_MOE_FP16, R3_MOE_AHALF) where the engine uses native FP8,
  and the layouts differ in which experts take which route. A per-layer diff of the two paths on the same inputs would find it.

Decision for now: the third helper's tail default stays off. 2 GiB passes cleanly but frees only 1.7 GiB of host pool; larger tails need
the path difference understood first. What the freed host pool buys: the KV cache is backed by the same pinned host pool ("the pool
backs 69% of the worst case ... a larger --host-pool-mib gives the cache more"), so host RAM freed from experts is context at full speed
once the engine hands it to the cache.

## Topology and card types as configuration, 2026-10-31

Owner requirement: card types are specified externally and their use configured, not hard coded. Done (details: `docs/reference/engines/radiance/LAYOUTS.md`):
`engines/radiance/archs.toml` (card types: plugin, formats, rows kernel, reserve, roles), `engines/radiance/layouts.toml` (named layouts over the host inventory),
`bigcherry layout list|show|env|card-env`, the launcher (`--layout`, `--budget ARCH=MIB`), the harness adapter (`R3_MOE_WORKER_LAYOUT`, or `R3_MOE_WORKER_DEVICES` with the card type taken
from the inventory), and the helper (no architecture whitelist, no backend table, no plugin map: `R3_MOE_WORKER_PLUGIN_FILE` and `R3_MOE_WORKER_FORMATS` are required;
the store-drain asm selects on the compiler's family macro `__GFX12__`). Replaced, not kept: `--tail-mib`, `FLASHNEXT_TAIL_MIB`, `XTX_ROCR`, `R9700_ROCR`, `RDNA2_ROCR`,
`lab_device_ordinals*`, `backend_for`, `gfx1100_backend`, `gfx1030_backend`, the helper's gfx1030 rows default. Not run on hardware yet: the launcher and a helper
start with the new settings (the GPU lease is held by the libr3 selftest); everything else is covered by unit tests and a three-architecture build.

## Why the 6900 XT does not speed things up, and where the lever is, 2026-10-31

Measured on the lane build of #472 (all four GPUs through `--layout`), natural text, 2048-token chunks.

**Share, not placement.** `flashnext-2xtx-rdna2-head` gives the 6900 XT one eleventh of the hottest experts (about 9% of routes) as a head helper. Prefill falls from 2951 to 1370 tok/s at 34K and from 2870 to 1269 at 147K; decode is level (127 to 119, 95 to 93). A chunk sends thousands of routes per layer to each helper, the 6900 XT takes about 0.0135 ms per route against the XTX's far lower cost, and every layer waits for the slowest helper. The share at which the 6900 XT finishes with the XTXs is below 1% of routes, which is what the cold tail already gives it. So tuning shares or placement cannot make the card pay; its cost per route has to fall.

**Tail at 4 GiB, context sweep** (`rad23-tail4g-ctx`): prefill 2951 / 2870 / 1950 / 1259 tok/s and decode 127 / 95 / 97 / 70 tok/s at 34K / 147K / 332K / 663K used context. The no-tail sweep of the same build line gave 3200 at 32K, 1854 at 288K, 1121 at 580K and 74 decode at 580K: the tail neither costs nor buys speed, at any context.

**The tile is not the bottleneck.** The libr2 f16 tile was an emulation (64 `ds_bpermute` and 128 FMAs). A standalone benchmark on the 6900 XT (`tile.hip`, CPU-checked): the emulation 4.2 T MAC/s; the same exchange with `v_dot2_f32_f16` 4.3 (LDS-exchange bound); a DPP form (neighbour swap for the upper row, `row_share` broadcast, dot2, no LDS) 5.45. The DPP form is now `libr2/r3_mma16.h`. In the real kernel it is bit-identical (same replay hashes) and 4% faster: the grouped GEMM takes 29.9 ms a call against 31.1 ms. The kernel runs at about 0.57 T MAC/s, about 10x below the tile, so the time is elsewhere, most likely dequantising the iq4 weights (each weight is used by about 32 routes). `__builtin_amdgcn_fdot2` gave wrong sums on gfx1030 in the benchmark; the instruction by inline asm is correct.

**GEMM against few-rows kernel at large lanes.** Replaying two 2048-row captures on the 6900 XT, the rows kernel took about 115 ms a request and the GEMM path 93 ms, so the GEMM path is already faster for large lanes. `R3_MOE_WORKER_ROWS` is a card-wide setting; the helper should pick by row count.

**Next lever.** An integer path on gfx1030: int8 activations, weights dequantised once to int8, `v_dot4_i32_i8` (22 T MAC/s measured, 5x the tile roof). That is a new expert kernel for libr2, to be confirmed first by a counter profile that shows the dequantisation dominating.

