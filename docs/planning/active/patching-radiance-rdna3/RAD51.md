---
id: RAD51
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-08T22:20:00+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P2
---

# Optional CPU BAR-pushed remote-MoE results, without CPU-driven engine scheduling

## Description

**Design-only and blocked on RAD50.** RAD42's GPU-pull/chunked CPU-combine
path already overlaps helper writes, the next CPU combine and the engine
GPU's previous-chunk read. RAD40 found a contended CPU-BAR bandwidth
advantage on this machine, but a CPU copy *after* combine may serialize
the CPU producer and lose against the already-overlapped baseline.
RAD51 is a candidate **opt-in serving integration** only if RAD50 proves
working CPU VRAM access, cross-process sharing, ordering and critical-path
benefit. Existing default remains unchanged. This is not an instruction
to make serving edits on the RAD50 experimental worktree.

RAD49's protocol v25 merged on main in PR #771
(`a139d293c`), after the RAD50 worktree branched. The implementation
branch **must** start freshly from current `origin/main`, not copy
v24 layout/slot indexing from the older experimental worktree.

## Steps

1. **Baseline and go/no-go.** Under the exclusive GPU lease, collect the
   same-session production trace of the chunked baseline: helper
   `chunks.seq` ready, combine wait+compute, `chunks.done`, GPU
   first read/last add and finish. Capture XTX0 host-read contention,
   request size, CPU occupancy, link throughput and p50/p95/p99. Feed
   measured values into RAD50's `critical_path_model.py`.
   **No serving implementation if the serial-copy design fails the
   critical-path budget or the interprocess mapping is unsupported.**
2. **Cross-process memory lifetime proof outside the engine.** Engine-like
   process allocates a per-consumer VRAM buffer. Helper-like CPU process
   imports it and writes through an actually CPU-accessible mapping,
   WITHOUT requiring another GPU's peer mapping. Prove close/restart,
   multiple consumers, failed attach, read-only/invalid handle and
   cleanup, along with full contents and generation identity. Both
   ROCr `hsa_amd_ipc_memory_create/attach` and HIP IPC may be tested,
   but HIP IPC can enable peer access and is NOT proof of CPU BAR mapping.
3. **Concurrent publication proof outside the engine.** CPU writes GPU
   VRAM, publishes an independent per-chunk ready generation, and the
   GPU already executing a wait observes readiness and reads all data.
   Verify correct happens-before across *VRAM and host-mapped flag
   memory*, including delayed posted writes, chunk reset, memory
   ordering, fault injection, missed wakeups and thousands of recycles.
   A serial kernel launch after CPU write is insufficient. If a CPU
   release fence plus a host-ready word does not guarantee visibility,
   select a supported visibility mechanism and remeasure its cost.
4. **Minimal engine-owned buffer pool and strict opt-in.** Add per-engine
   **consumer rank** fixed-address device buffer(s), allocated before
   the graph records their addresses and exported only after validation.
   All processes import and validate once at setup, never per step.
   Choose the transport only on prefill/combined large-lane steps, not
   decode/small lanes. The setting belongs in the validated serving
   configuration/manifest; a forced experimental mode must fail closed
   rather than silently return wrong data. Default remains mapped-host
   GPU pull.
5. **Helper 0 combine destination policy.** In `worker_combine.h`,
   reuse the existing large-lane `CombinePool` and per-rank
   `combine_index(consumer)` / `chunks[consumer]`. Compare (a)
   direct FP16 stores into CPU-visible GPU VRAM with (b) separate
   staging+CPU BAR copy, using real p95 latency. Direct output may
   avoid the serialized extra copy but its write-combining and
   non-temporal stores must be measured. Keep original host result
   planes intact for overflow/sentinel fallback and shadow checks.
6. **Consumer path.** In `moe_mailbox_ops.hip`, keep request ticket
   verification, rank selection, per-chunk polling owner and
   downstream tail wait/parent seal. For opted-in completed chunks,
   choose the engine VRAM combined-plane pointer rather than the mapped
   host pointer. Prove `kCombineFallback` still reads original
   per-helper BF16 planes in ascending helper order. Ordinary
   uncombined and decode passes use existing code.
7. **Quiescence and failure paths.** Extend close/admission/drain
   lifecycle in `moe_remote_hook.cpp` to cover buffer ownership,
   IPC attachments, in-flight CPU BAR writers and engine readers.
   No memory unmap/reallocate until last consumer/parent has retired,
   including placement epoch change and helper crash. Add deterministic
   fault/status handling and bounded timeouts, never indefinite
   GPU-spin or CPU spin. Changing the shared protocol requires
   a version bump and all parties rejecting a mixed version.
8. **Production A/B and quality gate.** Run both the one-rank/two-helper
   combine and the RAD49 TP2/two-helper layout under KL/quality gate,
   same payload and GPU lease, replay and 8-session large prefills,
   42K probe, 1/4/8 stream decode, and burst/idle power. Leave off
   by default; merge only if the observed benefit and correctness
   survive repeated paired runs. Separately consider CPU push of
   *activations* or TP all-reduce only with their own plan/gates.

## Detailed Solution & Technical Design

### Current control flow (main protocol v25)

- Engine `moe_mailbox_ops.hip`: `mailbox_publish_kernel` writes
  request/activation metadata and releases helper lane sequence;
  `mailbox_add_rows_kernel` waits on `h->chunks[rank].done[c]`
  using one poller per chunk and device-local mirrors. Its
  `combined_row` currently points into `g.map`, mapped host RAM.
  `mailbox_add_tail_wait_kernel` and `mailbox_add_finish_kernel`
  order final consumption. **Do not insert a blocking CPU callback
  into every layer/step.**
- Helpers `moe_remote_gpu.h` / `moe_lane_reduce.h`: GPU writers
  publish `chunks.seq[helper][chunk]` after system-scoped ordering.
  `worker_lanes.h` already hosts a large-lane CPU thread that calls
  `combine_step` without waiting for the entire helper GPU stream.
- CPU `worker_combine.h`: `CombinePool::combine` sums BF16
  contributors in helper order to FP16 with the exact sentinel
  `kCombineFallback` for overflow; uses per-consumer
  `combine_index(consumer)` and `chunks[consumer]` in v25.

### Alternative dataflow without CPU orchestrating GPU work

**Baseline:** helper GPU -> mapped host result -> CPU combines into
mapped host FP16 -> GPU pulls FP16 chunk -> add on GPU.

**Candidate:** helper GPU -> same mapped host result -> SAME CPU
combiner -> CPU writes preallocated consumer VRAM -> CPU-owned ready
generation -> GPU's already-queued add kernel reads locally.

There must be no new engine-host round trip per layer, no HIP stream
synchronization after each chunk and no dynamic HIP IPC attach on the
hot path. CPU readiness is already required for the current combine;
the *new* cost is VRAM write, cross-memory visibility, and any ready
publication delay, plus potential CPU worker contention. A separate
CPU copying thread may permit pipeline overlap but needs its own
ring, ACK/backpressure and wakeup costs: never assume it wins.

### Hard correctness invariants

1. Destination identified by `(consumer rank, ticket id, placement epoch,
   generation slot, combine generation, chunk)`; no stale ticket or
   second producer can publish the ready word.
2. Release of ready must prove all CPU writes to **that VRAM** are visible
   to the GPU, not merely that host RAM stores were ordered. Test
   heterogeneous memory ordering explicitly; `std::atomic_thread_fence`
   alone is not a proof for posted PCIe writes.
3. GPU must load/acquire the ready generation before reading FP16
   payload. One poller per chunk regardless of whether the first
   token has a helper route (RAD42 deadlock regression).
4. No buffer reuse/free/epoch transition until all actual
   consumer GPU readers have finished and the parent is retired.
   Helpers may finish their result write before GPU's local read.
5. Fallback token's first FP16 word `kCombineFallback` retains raw
   helper BF16 planes. No loss of shadow/reference results, NAN/
   overflow handling, helper order, or RAD49 exactly-once TP2 sum.
6. Failure must be explicit, bounded and never half-transition from
   BAR to host mode mid-step: select transport before publishing
   the request and keep it immutable through retirement.
7. Tests must include 1 rank + 2 helpers, TP2 + 1/2 helpers,
   no-helper-route first token, empty helper chunk, wrapped
   generation, helper death, CPU slow writer and shutdown races.

### File-level integration sketch (only on a future branch)

| Area | Proposed change | Risk |
| --- | --- | --- |
| `moe_remote_hook.cpp/.h` | Preallocate/export one stable VRAM combined-plane per consumer, binding and cleanup, attach validation | **Critical: cross-process CPU mapping/graph lifetime** |
| `moe_remote_shm.h` | Versioned ready/ack and transport/slot identity fields, no reuse until sealed | **Critical: correctness/deadlock** |
| `worker_combine.h` / `worker_lanes.h` | Direct CPU output target, measured write drain and ready publication; retain current pool | **High: CPU bottleneck and fallback** |
| `moe_mailbox_ops.hip` | GPU-local combined-row source, visible ready generation; preserve chunk poller and final tail wait | **High: synchronization, overflow fallback** |
| `moe_remote_gpu.h`, `moe_lane_reduce.h` | Initially **unchanged**; existing per-chunk GPU->host publication | Low if kept unchanged |
| Validated helper/launcher configuration | Opt-in only and route only qualified large combined steps | Medium: per-rank mismatch |
| Card-free C++ + GPU test suites | Protocol race/fault, crossprocess canary, KLD, baseline A/B | High coverage obligation |

### Risk ranking and stop/go

**P0 hard blockers:** CPU-accessible engine GPU VRAM shared with
a *separate helper CPU process* without P2P; visibility of posted BAR
writes before a GPU-visible ready word; producer/consumer lifetime,
fault and epoch handling. If unsupported, **do not implement**.

**P1 potential performance failures:** existing CPU-combine/GPU-pull
pipeline can overlap; adding serial CPU copy may *increase makespan*
despite higher BAR GB/s; direct FP16 stores to write-combined VRAM
may stall/slow the CPU combiner; CPU polling/wakeup/publish overhead
may dwarf a small decode payload; memory-traffic contention, CPU
core occupancy, GPU polling and MES power. Measure per-chunk and per
layer under real simultaneous load, not only synthetic BAR bandwidth.

**P2 integration risks:** TP2 rank consumer alternates by layer,
separate destination and flags per consumer; source pointer and
non-temporal GPU loads differ for VRAM; changing protocol/ABI in a
recorded graph requires stable pointers and compatible all-party
startup; explicit shadow and fallback compatibility.

On the experimental model with **assumptions** (256-row chunks, 2048
rows, 150us/chunk CPU combine and 100us/chunk producer interval),
RAD40 *measured* contended bandwidths 6.8 GB/s pull and 10.3
GB/s BAR push yield ~1932us baseline and ~2248us serialized
CPU-copy makespan (about 16% slower). These times are **not
observations** of serving; they demonstrate that link throughput
alone cannot justify serial-copy integration. The optimistic
direct-output case assumes *zero* extra CPU write cost and is
not a performance estimate.

Gate: >=15% reduction in affected transfer makespan or >=0.3ms
off p50 layer critical path, then >=2% end-to-end on qualified
prefill, <=5% p99 latency regression, KLD and correctness parity.
Decode should not regress at any supported configuration. No
default promotion without an explicit owner-approved decision.

## Code Samples & Guidance

RAD50 lab sources:
`tools/lab/rad42-bar-push/critical_path_model.py` (unmeasured
timing/sensitivity model), `cpu_handoff_bench.cpp` (host-only
lower bound, cannot model a GPU-originated futex wake),
`pool_inventory.cpp` (HSA capabilities) and
`bar_write_canary.hip` (serial CPU write/GPU verify),
`bar_ipc_canary.hip` (two-process HSA CPU attach), and
`bar_live_ready_canary.hip` (GPU polling before CPU writes).
Follow the evidence and gate journal in
`tools/lab/rad42-bar-push/PROGRESS.md`.

Do not replace GPU-driven `request_sequence` or tensor-parallel
all-reduce with CPU scheduling; neither is within RAD51's scope.
The presently used managed Radiance patches, not `vendor/radiance`,
own engine graph/plugin declaration changes.

## Files

**Future branch only:** `engines/radiance/components/moe-remote/`
files listed above, the validated serving configuration and its
tests, managed Radiance patch only if ABI changes require it.
**Current experiment:** only RAD50 lab files and this plan item.

## Validation

First RAD50 Gates 0–5 including both serial and concurrent GPU
canaries and actual cross-process mapping with documented lease.
Then card-free unit tests for version/identity, consumer rank,
missed chunk poller, no-route chunks, early ready, stale epoch,
timeout/fault, allocation failure and full teardown; compile matrix
for gfx1100 and gfx1201. Finally A/B parity and quantitative
serving runs under exclusive GPU lease, with before/after traces.

## Effort & Risk

L, advanced. Main uncertainties are heterogeneous memory visibility,
cross-process CPU mappings, CPU resource contention and whether the
old overlapped pipeline was already the faster one. A production
implementation is not justified until the critical blockers pass.

## Standards

Project-owned worktrees, no vendor source changes, strict manifest
validation, all-party protocol compatibility and GPU lease.
Keep no host-specific paths or hardware addresses in tracked code.
Honor the RAD49 and RAD42 test/quality gates.

## Acceptance Criteria

1. RAD50 verifies cross-process BAR mapping and concurrent
   GPU-ready publication under the real hardware/ROCm release.
2. Baseline trace and CPU-overhead sensitivity include overlaps,
   CPU work, per-chunk wait, GPU idle and p95/p99.
3. Zero stale/duplicate reads; exact TP2 and FP16/fallback outputs;
   crash and shutdown safety including multi-rank lifetime.
4. Opt-in serving mode defaults off and leaves decode unmodified.
5. Repeated same-session evidence meets documented end-to-end
   and critical-path thresholds without quality/latency regression.
6. Explicit owner approval and a separate reviewed PR before
   promoting this plan item from pending.

## Notes

Blocked by RAD50. Related RAD40, RAD42 (existing design B), RAD49
(merged v25 / TP2), and the existing no-P2P deployment constraints.
Do not use a historical v24 worktree as the production source.

### Current hard gate (2026-11-09)

RAD50 serial CPU BAR writes passed on both XTX cards, but the
**cross-process CPU writer remains blocked**. Legacy HSA CPU-agent
IPC attach fails with OUT_OF_RESOURCES; GPU-only IPC attach
succeeds but does not provide a writable CPU VMA, even after a
CPU access grant. GPU-local VMM DMA-BUF export also succeeds,
yet CPU `get_access` reports INVALID_AGENT. No helper CPU has
verified a GPU-local payload written into the engine's allocation.

**Do not implement the file-level BAR transport proposed here** until
a cross-process CPU mapping and GPU readback both pass. If mapping
is unavailable, a *separate* redesign may place CPU BAR writes in
the engine process and keep results in host shared memory; that
adds CPU orchestration/copying on the critical path and must be
measured against the already-overlapped GPU-pull baseline first.
No default engine change is justified by serial single-process data.

## Change Log

- 2026-11-08T22:20:00+00:00 (created-by): Proposed staged,
  opt-in production plan with risk ranking and measured-vs-modelled
  distinction. No serving code changes authorised or made.
