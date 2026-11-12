# RAD50 slice-by-slice optimization: measured hardware and gate plan

**Updated:** 2026-11-09. **Branch:** `exp/rad42-cpu-bar-push`.
**Scope:** lab-only DMA-BUF CPU-to-XTX VRAM remote-MoE transport, no serving changes.
**Evidence:** `evidence/2026-11-09/rad50-slices-*.log` and
`rad50-slices-validated-summary.txt`. Full pre-existing serial results:
`GATE3_RESULTS.md`, `GATE3_COMBINE_RESULTS.md`; first live failures:
`GATE4_LIVE_RESULTS.md`.

## Executive decision

1. The catastrophic 768-row live failures were **false deadline failures
   caused by use of `clock64()` on gfx1100**, not demonstrated FP16 data
   corruption. Device-local records showed `exit_reason=2` while the
   recorded clock delta was only ~853–1,113 ticks. Replacing the deadline
   with `wall_clock64()` scaled by the HIP-reported wall-clock rate let
   GPU-already-running tests complete correctly.
2. The original single-block full-payload GPU verifier is extremely
   inefficient (7.40 ms p50 for 768 rows) and masks actual transport
   latency. A separate GPU waiter followed by a 256-block parallel
   verifier and done kernel reduced 768-row diagnostic p50 to ~0.464 ms,
   **without reducing verification coverage**.
3. Both RX 7900 XTXs passed 100 repeat generations of a 768-row
   direct CPU FP16 combine into GPU-local DMA-BUF, with GPU kernel
   *armed before* each CPU write, full word-wise GPU verification,
   exact ready ticket, ACK and reuse. No P2P required. Both a
   parent-published coherent relay and child direct-published control
   path passed. The current data are not a portable memory ordering
   proof, fault-recovery proof or engine-inference result.
4. **CPU FP16 combine dominates now**. For 768 rows, its p50
   is ~363–364 us, around 78% of the ~464 us direct-live diagnostic.
   Optimize numerical CPU combine/data placement before chasing
   sub-microsecond DMA-BUF synchronization calls.

## Measured per-slice budget

The end-to-end diagnostic includes GPU launch/arming, remote CPU
combine + socket acknowledgement, and GPU visibility/full-payload
verification + done acknowledgement. Median component values are
independent quantiles and are therefore not exactly additive.

| Slice | XTX 0 p50 | XTX 1 p50 | Finding | Priority |
|---|---:|---:|---|---|
| GPU waiter launch + arm, 768 rows | 24.14 us | 24.06 us | Per-generation GPU scheduling overhead | High |
| CPU combine into VRAM, 768 x 2560 FP16 | 363.40 us | 363.62 us | Dominant cost, real production `combine_row_f16` on deterministic BF16 | Highest |
| Helper command + combine + ACK | 368.25 us | 368.99 us | Socket/ack beyond measured combine ~5 us | Medium |
| DMA-BUF SYNC start/end | ~0.3–0.4 us (observed warm samples) | ~0.3–0.4 us | Do not remove mandatory cache-coherency operation | Low |
| Post-ACK CPU/GPU completion, direct | 71.05 us | 70.52 us | Verifier, GPU kernel sequencing, host completion | High |
| Whole live test, direct, 768 rows | **463.87 us** | **463.71 us** | 100 verified generations; no engine inference claim | Baseline |
| Whole live test, parent relay, 768 rows | 469.06 us | 471.06 us | Difference only ~5–7 us; not same-session randomised A/B | Safety alternative |
| Original single-block live verifier, 768 rows | ~7404 us | not measured in identical run | Severe GPU verification bottleneck | Fixed in new lab mode |

Other direct-live p50 (30 generations):
1 row: XTX0 128.04 us, XTX1 118.55 us;
16 rows: 133.23 / 133.34 us;
64 rows: 156.44 / 156.77 us;
256 rows: 264.72 / 261.53 us.
At small sizes, fixed arm + GPU completion overhead overwhelms useful work.

**Tail caution:** 100-generation direct 768-row p95 596/590 us, p99
869/940 us, and max 1595/1490 us for XTX0/XTX1 respectively.
These are diagnostic distributions, not production p99/SLO results.
First-iteration cold-start is included, and the per-run samples are
neither stable workload traces nor statistically powered for p99.

## Ordered optimization slices, decision gates and tests

### S0 — Trustworthy timer and evidence (GATE: PASS on gfx1100)

- Remove all use of `clock64()` as a deadline on gfx11 canaries.
  Use `wall_clock64()` with `hipDeviceAttributeWallClockRate`, plus
  an independent host timeout. Keep CPU/GPU local diagnostics of
  `exit_reason`, `gpu_seen_ready`, `spin_loops` and mismatch count.
- Separate strict complete-run analysis from provisional observed
  samples. Long runs output first three, every tenth and final sample
  to stay within remote stdout capture limits, while full-run
  percentile vectors include **every** generation. Strict parser
  rejects absent post-cleanup marker, missing scheduled samples,
  malformed quantiles, mismatches or nonzero exit.
- **Next:** card-free negative tests for malformed logs, and a repeated
  100–200-generation smoke on both XTXs after any kernel change.

### S1 — CPU FP16 combine and destination placement

- **Baseline:** production `combine_row_f16` with deterministic
  normal/masked/overflow/NaN profiles. Preserve BF16 accumulation
  ordering, FP16 canonical NaN and `kCombineFallback` exactness.
- **Compare:** CPU host-mapped result, direct DMA-BUF output, optional
  staging + memcpy, aligned vs unaligned row offsets, various helper
  counts, row batches 1/16/64/256/768 and affinity/NUMA.
- **Optimize only measured bottlenecks:** AVX2/F16C instruction path,
  write-combining behavior, CPU-side scheduling, hot working set,
  parallel combine workers without cache-line collisions. Do not
  add an extra full-plane copy by default; earlier 4 MiB staged A/B
  gained only ~1.11x versus 1.36–1.41x synthetic direct.
- **Gate:** full-GPU bit parity across randomized edge cases and
  multiple generations; at least three paired same-card runs.
  Report CPU time, GB/s, p50/p95 and core utilization before selection.

### S2 — CPU BAR stores and DMA-BUF synchronization

- **Baseline:** 0.3–0.4 us warm SYNC ioctl pair in measured runs,
  subject to cache and kernel variation; no evidence it dominates.
- Separate compute versus stores with the existing direct/staged
  controls. Check cacheline alignment, memory placement and
  4 KiB–4 MiB scaling. Test contention only when GPU lease allows.
- **Gate:** never omit required DMA-BUF SYNC START/END, never publish
  ready before SYNC_END, and verify every GPU word after reuse.

### S3 — Interprocess command, publication and ACK

- Current persistent helper and Unix socket cost beyond
  `combine_row_f16` is ~5–6 us median for 768 rows. Parent coherent
  relay adds ~5–7 us full-path median versus direct child publication,
  **not** a proven same-session gain.
- Compare both mechanisms under identical payload and samples in
  alternating order, then test wrong generation, early publication,
  child death, truncated message, close/restart, timeout, duplicate
  ready and buffer reuse. Include KFD shutdown before unmapping.
- **Gate:** no stale generation, mismatch, unchecked FD lifetime or
  unbounded waiter. Retain relay and existing host-pull as fallbacks.

### S4 — GPU waiter and consumer

- **Verified isolation:** one GPU block verifying ~1.97 million FP16
  values took ~7.4 ms. The 256-block verifier plus separate waiter
  reduces the live 768-row diagnostic to ~464 us complete, in which
  only ~70–74 us is after helper ACK.
- Next benchmark: 1/32/64/128/256 verifier blocks, launch count and
  queue/stream behavior, GPU event kernel time vs host completion.
  Test a consumer that waits and *directly executes the actual fused
  Radiance gather/add*; the standalone full-word verifier is not
  representative of an inference kernel.
- **Gate:** exact every-word checks in the standalone mode; then
  production-equivalent numerical/once-only tests. Do not trade
  correctness for an apparently faster readiness polling loop.

### S5 — GPU dispatch/CPU orchestration

- The ~24–36 us to arm the running GPU and ~70 us after publication
  are large for tiny batches. Model whether a persistent GPU waiter
  and pre-enqueued producer/consumer streams remove launch costs,
  and whether a CPU wakeup or host roundtrip enters the decode path.
- **Gate:** benchmark under actual TP2 asynchronous overlap and
  multiple outstanding chunks; latency to first token and p95/p99
  must not regress versus current mapped-host GPU-pull.

### S6 — Multiple GPUs, contention, reliability and whole inference

- Benchmark both XTXs concurrently with CPU remote-MoE workers and
  controlled PCIe host-read traffic. Check NUMA affinity, PCIe
  pressure, GPU memory allocation lifetime, no unauthorized P2P,
  helper crash/restart, and per-rank once-only consumption.
- Feed measured stage distributions into `critical_path_model.py`
  with actual CPU-combine and Radiance gather times. Compare the
  existing overlapping GPU-pull pipeline, **not** only serialized
  microbenchmark totals.
- Then run paired, same-model/quant/seed, same-queue-depth TP2
  prefill/decode and KL-quality checks on dual XTX, including
  small and long prompts and p95/p99. Keep a shape-dependent opt-in
  fallback; 1-row CPU-push is not automatically faster.
- **Production consideration only:** sustained >=2% inference
  benefit with parity, no >5% p99 regression, and a demonstrated
  material layer critical-path saving under contention; thresholds
  are provisional review gates, not achieved measurements.

## Current recommendations

S0 and S4 lab-debug correctness fixes are ready for review as an
**experimental code checkpoint**, but not engine integration.
S1 (CPU combine throughput and locality) has the largest expected
return; S5 (GPU dispatch and host completion) is second.
S3 needs durability/safety before we can remove the parent relay.
S2 is not worth tuning in isolation until there is evidence SYNC
cost becomes material. S6 remains completely open as a serving-speed
claim. Maintain the original host-GPU-pull path for comparison and
fallback.

**GPU policy:** re-check `agent-lease status` and ROCm SMI, obtain
exclusive lease, preserve successful and failed logs, release only
a lease owned by the test session. RAD49 can use the GPUs at any
time; avoid competing with its work. No extra git worktrees.

## S4 follow-up: measured verifier-block sweep (2026-11-09)

The same RAD50 lab executable now supports 1/32/64/128/256 verifier blocks for direct-child and parent-relay GPU-live tests; default 256. It retains full-bit payload checks, explicit ready/done tickets and strict post-cleanup completion. **22 complete captured runs strictly passed; one interrupted local capture was rejected and retained as PARTIAL.** At 768 rows XTX0, 100-generation p50/p95 were 479.48/1698.41 us with 128 blocks versus 464.23/520.50 us with 256; a single block took 7.5 ms. Both cards favored 256 for post-ACK GPU/host completion, although XTX1's CPU-combine contention prevented a clean total-time comparison. One-row differences were small/inconsistent. Concurrent build activity prevents causal inference.

**No change to the 256-block default** and no S4 production-consumer DONE claim. Highest-value next slice is S1 CPU combine/destination, then failure/restart safety, real fused consumer and TP2 overlap. See [S4_VERIFIER_SWEEP.md](S4_VERIFIER_SWEEP.md). RAD50 GPU lease released and RAD49 now owns it.

## S1 card-free CPU combine sub-slice (2026-11-09)

A fixed single-tester CPU-only mode now measures real AVX2/F16C production combine into host RAM versus stage+host memcpy, with 1/2/4 helpers, hot versus unique source rows, exact scalar FP16 bit checks and bounded sample output. Strict log integrity tests and 106 Python unit tests pass. **Correctness/measurement harness complete, performance tuning NOT complete.** A 2-helper 768-row unique input varied from ~383 us to ~785 us between runs while unrelated ROCm builds saturated 98.55% of CPU. Do not select buffer layout, helper count, SIMD paths or CPU affinity on these unpaired, contended timings. Next: repeat on idle, affinity-controlled CPU with model-representative row heat; then exclusive GPU-lease BAR destination and overlapped TP2 tests. See [S1_CPU_COMBINE.md](S1_CPU_COMBINE.md).
