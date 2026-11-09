# RAD50: optimize each slice, then the real Radiance critical path

Status: 2026-11-09. Branch `exp/rad42-cpu-bar-push`. One existing
worktree. All changes so far are lab-only. No vendor, production-engine,
model weights, serving flags, new worktrees, pushes or PRs.

**Goal:** safely compare the CPU-only remote-MoE worker combining directly
into GPU-local DMA-BUF VRAM against RAD42's working path where the GPU
directly reads mapped host memory. This requires (1) correctness of *every*
stage, (2) stage-level optimization using repeatable measurements, and
(3) a final paired inference experiment with overlapping producer/consumer
schedules. A fast isolated memory transfer is never sufficient.

## Current critical findings — verified hardware

- Both RX 7900 XTXs (gfx1100, HIP 0/1) permit CPU-only cross-process
  DMA-BUF mmap/writes to GPU-local VMM allocations. Serial GPU bit checks
  and real production `combine_row_f16` pass. HSA CPU-agent IPC import
  failed but is **not** the selected transport.
- In prior 768-row serial AVX2/F16C CPU combine A/B, DMA-BUF direct output
  reduced median serial producer+GPU full-word check by about 79–81 us:
  XTX0 700.87→621.96 us, XTX1 703.44→622.16 us (100 iterations).
  This was an uncontended deterministic BF16 input, not serving.
- Original GPU-already-running test timed out at 768 rows (GPU saw ready=0).
  Diagnosis found an invalid deadline source: **HIP explicitly documents
  `clock64()` as not working properly on RDNA3/gfx11**. GPU-local
  diagnostics confirmed exit_reason=2 (clock deadline) after only ~528
  loads, with an inconsistent `clock64` delta.
- Fixed GPU canaries now use `wall_clock64()` and
  `hipDeviceAttributeWallClockRate` (kilohertz). A three-second device
  safety deadline is computed as `rate_khz * 3000` wall-clock ticks.
  Host-side waits and no-mismatch/no-stale-ticket checks remain intact.
  After this correction **both ordinary mapped-host and extended
  fine-grained registration pass 768 rows on XTX0** (10/10).
- Separating the *one-lane GPU wait kernel* from a 256-block parallel
  FP16 verifier and a subsequent done kernel removed the dominant lab
  GPU-consumer bottleneck. A coherent parent-relay proved correct;
  direct CPU-child publishing to the mapped-host control also proved
  correct on **both XTXs**, across 1/16/64/256/768 rows.
- Direct-publication, parallel-verifier complete 15-iteration sweeps:

  | Rows (2,560 FP16/row) | XTX0 p50 us | XTX1 p50 us |
  | ---: | ---: | ---: |
  | 1 | 129.46 | 120.17 |
  | 16 | 131.30 | 126.63 |
  | 64 | 158.08 | 157.94 |
  | 256 | 276.30 | 258.01 |
  | 768 | 590.78 | 506.28 |

  Per-run tails include initial cold iterations. Do not treat the
  15-iteration p99 as meaningful.
- Complete 100-iteration **compact** runs (first three, every tenth, and
  last generation emitted; every generation verified and included in
  the in-process percentile summary) at 768 rows:

  | Mode | XTX0 p50 / p95 us | XTX1 p50 / p95 us |
  | --- | ---: | ---: |
  | GPU ready published directly by CPU child; parallel verifier | 463.87 / 596.41 | 463.71 / 590.44 |
  | Ready relayed by parent into `hipHostMallocCoherent`; same verifier | 469.06 / 607.11 | 471.06 / 532.53 |

  Runs are sequential rather than randomized interleaved A/B. Direct
  publication avoids an extra notification hop and is the *simpler*
  candidate, but the median 5–7 us difference is **not a stable causal
  speedup estimate**. The parent relay remains a control/reference.
- The one-block live verifier with corrected timing took about 7.38 ms
  p50 for 768 rows on XTX0. The 256-block consumer-path experiments are
  about 0.46–0.59 ms, including the CPU combine. This large difference
  includes changes to GPU kernel structure/control flow and cannot be
  attributed solely to one CUDA/HIP API call.
- On the 100-iteration direct 768-row case the major stage budgets are:
  CPU FP16 combine about **363 us**, GPU arm about **24 us**,
  CPU-helper acknowledgement from armed about **368–369 us**, and
  host completion after ACK about **70–71 us**. These medians are
  **overlapping stage durations, not additive critical-path components**.
  DMA-BUF SYNC START+END is typically sub-microsecond after warmup.
- R9700/gfx1201 serial CPU BAR verification remains inconclusive.
  HIP's extended uncached host registration flag may not be available
  on Navi4X. Do not carry XTX-only observations to heterogeneous cards.

## Slice budget and optimization order

Each item needs four states: **correctness proof**, **baseline stage
measurement**, **one controlled optimization at a time**, **regression
and failure tests**. Mark a slice DONE only at all four, per GPU/profile.
Hardware and software support must be tracked separately.

| Priority / slice | Current evidence | Next optimization / isolation experiment | Gate |
| --- | --- | --- | --- |
| P0. Host/GPU clock and watchdog | GFX11 `clock64` caused false GPU-ready timeouts; wall clock fixes observed 768-row failures | Retain explicit GPU wall-rate checks, host deadline, timeout reason, and a card-free guard against regression; validate gfx1201 separately | No early success, no indefinite waiting, all stalls fail safely |
| P0. Exact ready/done protocol | Direct child publish and coherent parent relay each pass at 768 rows on both XTXs | One slot, many generations; intentionally stale/duplicate tickets, delayed/aborted helper, close/reopen, published-before-data negative canary; system-scope semantics and cache policy review | No false ready, lost generation, double-consume, stale data or use-after-free |
| P1. GPU consumer | One-block verifier was catastrophically inefficient; 256-block ordered consumer is faster | Profile waiter, verifier, done kernel separately; test stream graph capture, fewer launch boundaries and production fused mailbox consumer; avoid grid-wide spins or unsafe inter-block barriers | Full bitwise parity; no deadlock; kernel occupancy/launch measured |
| P1. CPU BF16→FP16 combine | Production AVX2/F16C output to VRAM ~363 us/768 rows, dominating measured path | Run real captured helper row heat/masks and multi-helper counts; compare AVX2 throughput, worker affinity/NUMA, store alignment, per-row vs chunk combine, direct versus staged; test any NT/write-combining store in **lab only** with required fences | FP16 bit parity including masks, overflow sentinel, NaN, signed zero, subnormals, ranks |
| P1. GPU-local CPU BAR stores | Direct stores beat an extra host-stage+copy; VRAM output costs ~50 us more CPU at 768 rows than host RAM | Isolate destination-store bandwidth from arithmetic; test alignment, page/NUMA placement, vector widths, cache policies and write batching with verified GPU readback | No torn row; no silent cache/ordering hazard |
| P1. Interprocess handoff | Persistent helper, SCM_RIGHTS only at setup, socket ACK per generation; arm ~24 us and ACK overhead around 5 us warmed | Separate GPU arm/host wake, child recv/send, parent ACK relay; test persistent ring slots, eventfd/futex and batch only when topology permits; avoid per-chunk spawn/map | Exact once semantics, bounded latency, recover on child exit |
| P1. DMA-BUF SYNC & fences | Two Linux SYNC ioctls per payload, usually ~0.3–1 us warmed on these XTXs | Time START vs END and CPU sfence independently under contention; batch only at legal ownership boundaries, no illegal sync suppression | Correct CPU-to-GPU visibility; no concurrent read during unfinished CPU writes |
| P2. Resource lifecycle | Single persistent FD/map and serial buffer reuse verified; crash path not proven | Export/import/close/unmap/restart/error injection; many sessions, KFD process cleanup, GPU allocation fragmentation and cache pressure | No stale FD, leaks, GPU hangs, or KV-cache capacity regressions |
| P2. Heterogeneous/multi-XTX PCIe | XTX 0/1 mapped; no P2P, prior contended RAD40 numbers are historical | Repeat tests with controlled host-read load on other XTX, CPU memory/NUMA pressure and all card topologies; separately investigate R9700 | No latency cliff, starvation or hidden P2P requirement |
| P3. Real Radiance TP2 pipeline | Serial and live lab verifier are not production fused add/gather | Replace verifier with actual fused remote-MoE consumer; workload-trace replay, branch-gated opt-in A/B, KL/output parity, prefill and decode with matched queues and thermal load | End-to-end benefit with bounded p95/p99, no correctness or production regressions |

## Measurement design: isolate, then integrate

**Stage telemetry required per generation** (with a common ticket):
`generation,device,pci,rows,fp16_bytes,masked_helpers,num_helpers,
cpu_wake_us,combine_us,bar_store_us,dma_start_us,dma_end_us,
host_publish_us,gpu_arm_us,gpu_wait_us,gpu_load_or_fused_add_us,
gpu_done_us,cpu_ack_us,total_us,errors,timeout_reason`.
For coherent parent relay, label that owner distinctly; do not call
post-ACK completion time "GPU execution time" because it contains
host synchronization and possible scheduling delays. Component
percentiles cannot be added naively.

**A/B design:** both XTXs; 1,16,64,256,768 rows; 10 quick correctness
generations then 100+ measured generations; performance `-O2` and
record executable SHA, ROCm version, CPU/GPU clocks, package temperature,
CPU core/NUMA and PCIe lane mode. Alternate A/B order and replicate
whole sessions. For 100+ generations use compact per-sample stdout,
preserving every failure, summary `emitted` count and a post-cleanup
completion marker. Keep **all partial logs** and diagnose them as
provisional rather than suppressing. Test idle and controlled host-read
contention separately. GPU tests require a fresh exclusive lease,
a no-foreign-KFD-PID preflight and explicit release by the task that acquired
it; never preempt RAD49 or another owner.

**Critical-path accounting:** prior RAD42 production overlaps CPU
combine for chunk i+1 with GPU pull/merge for chunk i. The correct
model is two overlapping resources (CPU and GPU), not a sum of serial
stage medians. Use `critical_path_model.py` with recorded chunk sizes,
combine times, notification costs and actual fused consume times.
Include queueing, ring reuse, expert heat, remote worker load,
nonuniform placement, and TP2 rank once-only behavior. Maintain
maximum GPU-local headroom for model layers and KV cache. Compare
**per-layer critical path** and **whole-model** throughput/latency.
Fallback to existing mapped-host GPU-pull at small or unfavourable
shapes; do not hard-code the 16-row threshold until real replay passes.

## Stop/go sequence

1. **P0 correctness**: complete direct and relay live controls across
   both XTXs, 100+ generations; stale/duplicate/fault/timeout tests,
   successful restart and cleanup. Wall-clock safety regression test
   must fail if gfx11 `clock64` is reintroduced. No production code
   yet.
2. **P1 optimization**: one change per slice, with phase-by-phase
   counters and meaningful paired CIs. Verify each shape and both
   GPUs. Only then select direct vs relay, polling vs queued GPU work,
   chunk size and store mode.
3. **P2 load/heterogeneity**: GPU0/GPU1 simultaneous host-read
   contention; R9700 separately; VRAM fragmentation, CPU NUMA,
   model/graph reuse and helper failure.
4. **Whole-model gate**: actual Radiance remote MoE + TP2 with
   model-representative inputs, numerical/KL equality, prefill/decode
   and realistic queue depth. Provisional decision target: at least
   2% paired end-to-end gain without more than 5% p99 regression.
   If there is no sustainable real gain, keep the original transport.
   A 1.1×–1.4× **lab** ratio is not an inference speedup.

## Evidence and implementation

- New modes in the **one existing** `rad50-tester`:
  `dmabuf-live` / `dmabuf-live-fg` (clock-corrected original),
  `dmabuf-live-relay` (coherent parent publication),
  `dmabuf-live-parallel` (CPU helper publishes directly; same
  parallel 256-block GPU verifier).
- Sources: `live_control.hip.h`, `live_rounds.hip.h`,
  `live_relay.hip.h`, `bar_dmabuf_ab.hip`, `combine_patterns.hip.h`.
- Analyzer: `slice_analyze.py` strict versus `--provisional`;
  compact 100-generation logs distinguish emitted lines from all
  verified generations. Supporting previous results in
  `GATE3_COMBINE_RESULTS.md` and `GATE4_LIVE_RESULTS.md`.
- Raw successful, failed, partial and compact logs are in
  `tools/lab/rad42-bar-push/evidence/2026-11-09/rad50-slices-*.log`.
  Original build output remains under the external build root.

Upstream device semantics:
- ROCm HIP language extensions:
  https://rocm.docs.amd.com/projects/HIP/en/docs-10.1.0/how-to/hip_cpp_language_extensions.html
  (RDNA3 `clock64()` caveat, documented `wall_clock64()` + clock-rate attribute).
- ROCm HIP coherence:
  https://rocm.docs.amd.com/projects/HIP/en/docs-10.1.0/how-to/hip_runtime_api/memory_management/coherence_control.html
  (fine/coarse-grained host visibility and synchronization).
- Linux DMA-BUF:
  https://docs.kernel.org/driver-api/dma-buf.html
  (`DMA_BUF_IOCTL_SYNC` is cache coherency, not the whole
  application-level producer/consumer protocol).

## S4 update - 2026-11-09 verifier-grid sweep

Existing RAD50 tester accepts bounded `--verifier-blocks 1|32|64|128|256` for live direct and relay modes; 256 remains the default. Strict log parsing enforces matching block-count metadata. Both gfx1100 XTXs passed **22 complete captured full-word and ticket verification runs**; one interrupted local capture remains explicitly partial. At 768 rows on XTX0, 100-generation p50/p95: 256 blocks 464.23/520.50 us, versus 128 blocks 479.48/1698.41 us; a one-block verifier took ~7.5 ms. Both cards had faster post-ACK completion with 256 blocks, while XTX1's higher CPU-combine time confounded total comparisons. These runs overlapped another build; no causal/production speedup claim is warranted.

**Retain 256-block baseline.** S4 fused GPU consumer, S0/S3 failure lifecycle and S6 paired TP2 serving gates are open. S1 production CPU FP16 combine plus destination-placement isolation is next priority. Raw hardware logs, SHA, test status and limitations: `tools/lab/rad42-bar-push/S4_VERIFIER_SWEEP.md`. The RAD50 exclusive GPU lease was released; RAD49 holds the GPUs.
