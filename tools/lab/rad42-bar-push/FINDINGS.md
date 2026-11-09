# RAD50 preparation findings — risk review (2026-11-09)

## Conclusion

**Do not introduce per-layer CPU orchestration.** The current engine already
submits GPU work and GPU-side waits on per-chunk generations. The helper's
*large lane* has an existing CPU thread that reads GPU-produced host-mapped
chunk markers and combines them; only the combined-plane **destination**
is a candidate for BAR push. The helper's GPU kernels, engine routing and
tensor-parallel AllReduce should not be rescheduled through the CPU.

The expensive part may be **introducing another serial CPU copy** into
an already-overlapped pipeline. CPU push bandwidth can beat GPU pull
*under contention* while total layer-step makespan becomes worse.

## Source audit: current upstream versus experimental base

The experimental branch started from `6f7de471a`; current
`origin/main` includes RAD49 PR #771 (`a139d293c`, protocol v25),
so these source references must be revalidated before *production*
edits. Read-only verification on `origin/main` showed:

| Component | Existing activity | Proposed intervention |
| --- | --- | --- |
| `moe_mailbox_ops.hip`: `mailbox_publish_kernel` | Engine GPU publishes routes and request, release-scoped | **Unchanged** |
| `moe_lane_reduce.h`: `lane_chunk_publish` | Helper GPU writes host result and publishes chunk-ready sequence | **Initially unchanged** |
| `worker_lanes.h`: `run_large` | CPU large-lane thread already calls `combine_step` while helper GPU runs | **Reuse; no new CPU model scheduler** |
| `worker_combine.h`: `CombinePool::combine` | CPU combines selected BF16 helper rows to FP16 into host result plane, then publishes `chunks[consumer].done` | **Candidate output VRAM buffer; no extra producer thread unless justified** |
| `moe_mailbox_ops.hip`: `mailbox_add_rows_kernel` | GPU chunk poller mirrors one host ready flag into device RAM per chunk, then pulls host FP16 combined row | **Optional local VRAM pointer and separately proven ready condition** |
| `moe_mailbox_ops.hip`: tail/finish | Full helper synchronization and per-parent retirement | **Keep intact; must cover BAR buffer lifetime** |
| `moe_remote_hook.cpp/.h` | Engine rank-specific mapped host segment and device allocations | **Future per-consumer engine VRAM allocation; bounded IPC export and teardown** |
| `moe_remote_shm.h` | Per-consumer `combine_index(consumer)`, `chunks[consumer]`, `slot_of(helper,consumer,world)`, `consumer_of(layer,world)` | **Honor v25 rank/epoch/slot ownership** |

**Existing GPU wait bug:** RAD42 formerly hung because the first token
in a chunk did not poll `chunks.done` when that token had no helper
routes. The fixed first-block poll must survive any BAR change. No GPU
should read a new ready word without validating the correct generation.

## Risk ordering

1. **Critical, not yet proven:** HSA GPU-local memory CPU addressability
   and export to a separate helper CPU process on a no-P2P host. HSA
   capability query alone cannot prove a working CPU mapping. HIP IPC
   can request GPU peer access; that is *not* the intended transport.
   Prepared `bar_ipc_canary.hip` uses an HSA GPU-memory export and
   CPU-only child attach, and fails rather than falling back.
2. **Critical:** host CPU posted writes into GPU VRAM must be GPU-visible
   *before* the host-mapped ready generation is observed by an already
   running GPU kernel. A serial `hipDeviceSynchronize`/kernel
   launch can accidentally provide ordering; success there does NOT
   prove the concurrent protocol. CPU fences might not be enough.
3. **Critical:** GPU consumer may still be reading old buffer when
   CPU writer starts new epoch; buffer ownership must include ticket,
   rank, placement slot, generation and chunk, and remain stable
   through all downstream readers/parent retirement.
4. **High:** CPU combines and GPU pulls the next chunks **concurrently**
   today. Adding a CPU BAR copy *after* combine may serialize the
   producer and move link traffic onto CPU critical path. Direct FP16
   writes into VRAM may avoid the copy but the CPU AVX2/F16C stores
   might be much slower into write-combined/device memory than DRAM.
5. **High:** CPU wake/scheduling and extra GPU polling. The GPU already
   publishes requests; the existing large-lane CPU spins inside a burst
   and cannot be woken by GPU futex. Avoid host callbacks in every
   decode step, extra HIP sync, per-chunk thread creation, or per-step
   IPC import. A host-only benchmark is not a GPU wakeup measurement.
6. **High:** sentinel and fallback. A finite FP16 overflow is encoded as
   `kCombineFallback` in the first combined row value; the GPU then
   must read original BF16 per-helper host planes, not the new VRAM
   combined payload, and maintain helper order and exact output.
7. **High:** RAD49 TP2 alternates one consumer rank by layer. Per-rank
   buffers and ready words must match `consumer_of(layer,world)` and
   `slot_of(helper,rank,world)`. Other rank must not add helper row.
8. **Medium:** wrong buffer policy chosen during graph replay or
   placement; original host path must remain fixed and available;
   protocol version validation, startup rollback and shutdown need
   tests. No live transport selection mid-request.

## Link bandwidth versus existing overlap

RAD40 measured XTX1's *contended* mapped-host GPU pull at 6.8 GB/s
and CPU BAR push at 10.3 GB/s. A 256-row chunk at hidden=2,560,
FP16=2 bytes carries 1,310,720 bytes, so a pure bandwidth
model gives approximately 193 us GPU pull and 127 us CPU push.
**Those are transfer-only quotients, not layer-step timings.**

Example scenario in `critical_path_model.py` using *illustrative*
unmeasured times: GPU-produced chunks 100 us apart; CPU combine
150 us/chunk; GPU add 30 us/chunk; 2048 rows (eight chunks).
Current overlapped CPU combine + GPU pull makespan ~1932 us,
while a serial CPU copy after each combine makespan ~2248 us,
**16.4% slower** despite higher CPU BAR bandwidth. An optimistic
direct-VRAM-output model assumes zero added CPU store/publish cost
and therefore cannot yet support a performance claim.
Run `critical_path_model.py --help` and replace every assumed
parameter with actual traces once a GPU lease is available.

## Next tests, bounded order

1. HSA pool inventory and serial 4KiB/4MiB CPU write/GPU verify
   (`pool_inventory`, `bar_write_canary`).
2. Same test from *separate CPU-only helper process* through HSA IPC
   (`bar_ipc_canary`). If this fails, stop this architecture.
3. Same-payload CPU BAR vs GPU mapped-host pull A/B; profile CPU
   stores and p50/p95 under RAD40 contention (`bar_transfer_ab`).
4. Dedicated **live GPU waiter** protocol correctness and scheduling
   latency; no serial launch after data publication.
5. Actual FP16 combined rows and KLD/TP2 reference; then only
   consider the future opt-in production plan `RAD51`.

**Status (2026-11-09):** Gate 1 HSA discovery PASS (7 GPU-local CPU-accessible pools). Gate 2 serial CPU write → GPU read PASS on both RX 7900 XTX GPUs: 4 KiB and 4 MiB, 12 generations each, all words verified. The R9700 gfx1201 executable exited 1 with no diagnostics: its result is **inconclusive** pending a targeted rerun. Serial correctness does not establish cross-process access, concurrent visibility, CPU overhead or throughput; all remain unverified. RAD50 released its GPU lease.

**Later R9700 diagnosis:** HSA allocation, CPU access, 4 KiB stores and HIP memset all complete; execution stops before the first verification `hipLaunchKernelGGL` returns. This could be a launch stall/fault or the command's execution limit; **no GPU read was verified**. A separate HIP-only launch smoke mode is compiled but not run. XTX cross-process IPC remains unrun because `bar_ipc_canary-gfx1100` is not allowlisted. RAD50 released its own lease; HD2701 subsequently acquired the exclusive GPU lease.

## 2026-11-09 — cross-process sharing results

- XTX0/XTX1 serial, *same-process* CPU BAR stores to GPU-local HSA
  allocation remain verified (4 KiB/4 MiB, 12 generations each).
- XTX0 legacy HSA IPC **CPU-agent attach fails 4104 OUT_OF_RESOURCES**
  even after the exporter grants CPU access.
- XTX0/XTX1 HSA IPC **same-GPU-agent attach succeeds**. On XTX0,
  imported pointer info reports a hostBaseAddress and a later CPU grant
  succeeds, but the address is **not a writable CPU VMA**. The guarded
  4 KiB test refused to write; no parent GPU payload verification.
- New HSA VMM-capability probe on XTX0: GPU-local handle create,
  DMA-BUF export, reserve/map, CPU/GPU set_access and GPU get_access
  all succeed. CPU get_access fails **4100 INVALID_AGENT**. Linux
  reports a writable virtual region but the HSA CPU access proof
  fails, so **no CPU stores or cross-process VMM import attempted**.
- On R9700, HIP-only kernel dispatch and readback **PASS**, but the
  serial HSA BAR verification still stops at GPU kernel launch.
- **Architectural decision:** Direct CPU BAR push by a *separate*
  helper process is NOT currently demonstrated. Never promote RAD51
  or run throughput tests using GPU-only IPC as a substitute for CPU
  access. Possibilities to examine separately: a properly supported
  GPU-local DMA-BUF CPU mmap mechanism, or relocating the CPU combine
  destination/write into the engine process. The latter adds
  orchestration and may erase existing CPU combine/GPU pull overlap;
  model and trace it before any production change.
