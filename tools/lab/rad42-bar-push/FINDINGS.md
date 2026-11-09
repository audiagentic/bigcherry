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

**Status (2026-11-09):** Gate 1 HSA discovery PASS: 7 GPU-local CPU-accessible candidate pools across the four GPUs. Gate 2 GPU correctness canaries are compiled but their executable names are still blocked by the remote allowlist, so **no CPU BAR writes or GPU verification have run**. IPC, transfer performance and concurrent readiness remain unverified. RAD50 released its lease after this block; no GPU lease is held by the experiment.
