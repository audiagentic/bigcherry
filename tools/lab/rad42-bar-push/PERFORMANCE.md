# RAD50 Gate 3 — paired speed comparison

Status (2026-11-09): **completed measured synthetic serial Gate 3 A/B** on both gfx1100 XTXs, 5 sizes, direct/staged CPU writes, 30 iterations plus warmups per case. All data verified on GPU, exit 0. Direct CPU-push at 4 MiB: XTX0 1.414x and XTX1 1.362x over mapped-host GPU-pull; at 4 KiB, push loses ~2%. These are **not inference speedups**. See [GATE3_RESULTS.md](GATE3_RESULTS.md). 39 CLI checks and 50 card-free Python tests pass.

## Baseline versus candidate

The existing RAD42 consumer GPU **directly reads mapped host results**.
The correct baseline is *not* a GPU memcpy-to-VRAM implementation.

- Baseline: independent CPU-only child writes a host shared-memory file
  (memfd); the GPU parent registers those pages with hipHostRegisterMapped.
  The GPU checks every word by directly loading the mapped-host pointer.
- Candidate: the **same persistent child** writes a DMA-BUF mmap of GPU-local
  VMM memory exported by the parent. Each CPU access is bracketed by
  DMA_BUF_IOCTL_SYNC START/END, both timed. GPU checks every word by loading
  the GPU-local VMM pointer.
- Both paths use the same deterministic input pattern, GPU verification
  kernel, process/socket protocol, sizes, repetitions, and PCI identity.
  Process startup, memory allocation and mapping happen only once per run.
- A/B order alternates on each iteration. Three warmups precede each
  reported set. The result is serial CPU producer -> GPU reader, NOT
  concurrent readiness or an engine-layer benchmark.

## Inputs and outputs

The new mode is dmabuf-ab. It requires --device, --iterations (10..200),
--mode (direct|staged), and --lease-confirmed.

For each path, every word is GPU-verified over 4 KiB, 64 KiB, 256 KiB,
1 MiB and 4 MiB, at changing generations.

CPU writing styles:
- direct: computes each synthetic word and directly stores it into host
  RAM or GPU VRAM. The measured CPU write time includes computation.
- staged: prepares the same data in ordinary RAM, then performs memcpy
  to the destination. Reports preparation and memcpy separately.

To keep each remote-shell invocation bounded (the shell currently has a
short execution limit), the optional --bytes selector restricts an A/B run to
exactly one approved payload: 4096, 65536, 262144, 1048576 or 4194304.
With no --bytes argument the runner tests all five sizes. The selector is
accepted only for dmabuf-ab and never accepts an arbitrary allocation size.
Use --bytes 4096 --iterations 10 for the first correctness/performance smoke,
then move to 30 samples and larger payloads. Do not publish p99 based on
a 10-sample smoke as meaningful tail latency.

For each device/size/style/path the program emits RAD50_AB records:
prepare, CPU write, SYNC START/END, child total, parent socket roundtrip,
GPU kernel event, host GPU completion and serial producer+consumer
p50/p95/p99, plus approximate serial GB/s. RAD50_AB_COMPARE records
compute baseline/push ratio and time difference. A separate card-free
ab_analyze.py rejects unpaired or malformed output and compares pairs.
Always preserve raw stdout, stderr, command arguments and exit status.

## Runbook

Only after a verified-free GPU lease and no serving/KFD processes:

1. Rebuild the **same** rad50-tester with OPT=-O2 (currently compiled at
   OPT=-O1 for correctness preparation). Record commit SHA, ROCm version,
   compiler flags and device PCI addresses. Do not interfere with
   RAD49 measurements while its lease is held.
2. Acquire the exclusive gpu lease before any hardware run.
3. Run the following exact commands from the RAD50 worktree, with T set
   to the allowed absolute rad50-tester path:

    # First bounded smoke after acquiring lease:
    $T --test dmabuf-ab --device 0 --iterations 10 --mode direct --bytes 4096 --lease-confirmed
    $T --test dmabuf-ab --device 1 --iterations 10 --mode direct --bytes 4096 --lease-confirmed

    # After both smoke runs finish with full-word GPU correctness:
    $T --test dmabuf-ab --device 0 --iterations 30 --mode direct --lease-confirmed
    $T --test dmabuf-ab --device 0 --iterations 30 --mode staged --lease-confirmed
    $T --test dmabuf-ab --device 1 --iterations 30 --mode direct --lease-confirmed
    $T --test dmabuf-ab --device 1 --iterations 30 --mode staged --lease-confirmed

4. Save the output via the approved host's command capture. Analyze
   captured logs card-free using:

    python3 tools/lab/rad42-bar-push/ab_analyze.py /path/to/raw-log.txt

5. Release the lease and confirm status.

The --lease-confirmed argument is only an operator acknowledgement,
not a lease acquisition or ownership check.

## Optimization, controls, and decision

- Any wrong word, failed HIP host registration, DMA-BUF map/sync error,
  device mismatch, child failure or timeout STOPs the run.
- First determine which cost dominates: direct/staged CPU stores,
  DMA-BUF SYNC calls, IPC handoff, or GPU read. Optimize the measured
  bottleneck rather than guessing. Potential follow-ups include
  reuse/ack batching, SIMD/non-temporal stores, persistent combine
  threads, chunk sizing and CPU core/NUMA affinity. Keep mandatory
  DMA-BUF synchronization for correct cache coherency.
- Next, compare under controlled XTX host-read contention using an
  independent, reviewed workload under the **same exclusive lease**.
  This first version measures isolated performance only. Historical
  RAD40 contention data are not a current benchmark result.
- Feed actual stage timings and CPU combine traces to
  critical_path_model.py. The existing RAD42 baseline overlaps CPU
  combine with the previous GPU read: copying into VRAM after combine
  can regress even if BAR transfer alone is faster.
- Before opting RAD51 into serving, still require multiple generations,
  buffer reuse, crash/restart safety, GPU-already-running readiness,
  real TP2 MoE data, end-to-end paired KL/prefill and p95/p99 checks.

Proposed gates (not demonstrated): material transfer/critical-path
improvement (initial target >=15% or >=0.3ms/layer), >=2% end-to-end
inference benefit, and <=5% p99 regression. A serial microbenchmark
alone cannot justify production enablement.

Reference: AMD HIP documents hipHostRegisterMapped and
hipHostGetDevicePointer; the Linux kernel's DMA-BUF documentation
requires SYNC START/END and separately says the caller handles
GPU/device synchronization.

## Partial logs and durable evidence

Do not discard data when a run fails or times out. Save the raw stdout, stderr, command, exit status, timeout and truncation flags. The default ab_analyze.py parser requires a successful post-cleanup RAD50_AB_COMPLETE marker for final paired comparisons. Use `python3 tools/lab/rad42-bar-push/ab_analyze.py --provisional <log>` to inspect observed rows, incomplete pairs and diagnostics without claiming a verified run. The first complete and intermediate logs are checked into `tools/lab/rad42-bar-push/evidence/2026-11-09/` with the measured findings in [GATE3_RESULTS.md](GATE3_RESULTS.md).
