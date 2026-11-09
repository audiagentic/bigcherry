# RAD50 / RAD51 — hardware gate and critical-path runbook

This document is **instructions for a future leased hardware session**.
The experimental branch has not run any GPU benchmark or BAR canary.
Do not run the steps out of order, and do not terminate a lease owner.

## A. Preflight and measurement identity

1. Use the RAD50 experimental worktree, not primary main. Record exact commit
   and `origin/main` identity. Do not merge or rebase while a run is in flight.
2. `agent-lease status` must report no other GPU lease. Confirm
   `rocm-smi --showuse --showmemuse --showpids` has no foreign processes
   across all four GPUs. If a server, helper or lease remains, STOP.
3. Acquire the project-exclusive `gpu` lease using `agent-lease acquire
   gpu --ttl 2h --purpose "RAD50 GPU-local CPU write and IPC gate"`.
   Record owner/time. Run only the two relevant XTX GPUs initially;
   **the lease is still exclusive across the whole host**.
4. Record `hipconfig --full`, `rocminfo` (selected lines), kernel version,
   GPU PCI identities, driver, CPU affinity/NUMA node and negotiated link.
   Do not rely on a generic ordinal; the canaries match physical PCI IDs.
5. Use named evidence under the project work root, keyed to RAD50 and this
   source commit. Preserve command, binary SHA256, exit code, STDERR/STDOUT,
   GPU processes, p50/p95 and measurement window for each arm.
6. Never touch an active Radiance server. If a test has a GPU hang or
   incorrect bytes, STOP, release only your lease and preserve evidence.

## B. Build-only preparation (no GPU execution)

**Preferred new interface:** [TESTER.md](TESTER.md) documents a compiled-in,
single `rad50-tester` executable. Build it with
`make -C tools/lab/rad42-bar-push OUT_DIR="$BC_WORK_ROOT/builds/rad42-bar-push" rad50-tester`.
The exact name `rad50-tester` is currently allowed; use its fixed `--test`
modes with `--lease-confirmed` when an approved exclusive GPU lease is
actually held. This acknowledgement does not acquire or verify the lease.
Use `--self-check` and `--list` (CPU-only) after permission is granted.
Existing separate commands below document their original implementations;
there is no need to authorise each standalone executable.


From the *experimental worktree root*, after loading the project's
environment, compile into the project work-root build folder:

```sh
source tools/env/bigcherry-env.sh
mkdir -p "$BC_WORK_ROOT/builds/rad42-bar-push"

python3 -m unittest discover -s tools/lab/rad42-bar-push -p 'test_*.py' -v
python3 tools/lab/rad42-bar-push/critical_path_model.py --rows 2048
g++ -O2 -std=c++17 -pthread -Wall -Wextra \
  tools/lab/rad42-bar-push/cpu_handoff_bench.cpp \
  -o "$BC_WORK_ROOT/builds/rad42-bar-push/cpu_handoff_bench"
g++ -O2 -std=c++17 -Wall -Wextra \
  -I engines/radiance/components/moe-remote \
  tools/lab/rad42-bar-push/combine_reference_test.cpp \
  -o "$BC_WORK_ROOT/builds/rad42-bar-push/combine_reference_test"
g++ -O2 -std=c++17 -Wall -Wextra \
  -I"$ROCM_PATH/include" tools/lab/rad42-bar-push/pool_inventory.cpp \
  -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" -lhsa-runtime64 \
  -o "$BC_WORK_ROOT/builds/rad42-bar-push/pool_inventory"
for arch in gfx1100 gfx1201; do
  for unit in bar_write_canary bar_ipc_canary bar_transfer_ab bar_live_ready_canary; do
    hipcc -O2 -std=c++17 --offload-arch="$arch" \
      "tools/lab/rad42-bar-push/$unit.hip" -lhsa-runtime64 \
      -o "$BC_WORK_ROOT/builds/rad42-bar-push/$unit-$arch"
  done
done
```

The C++ reference test can only run under an execution environment
that explicitly permits running the locally compiled test binary.
A successful compilation is not an executed correctness test.

## C. Strict hardware gate order (lease required throughout)

1. **Gate 1 inventory:** Run `pool_inventory` and record GPU BDF, HSA
   pool location (GPU), allocation permissions and CPU access. If no
   CPU-accessible GPU-local pool exists, record FAIL and stop.
2. **Gate 2 single-process serial canary:** Run the architecture-matched
   `bar_write_canary --device HIP_ORDINAL` on GPU 0 and GPU 1, then
   optionally R9700. Requires PASS for both 4 KiB and 4 MiB, 12
   generations, 0 mismatches. A coherent kernel launched after the
   CPU stores might internally impose a flush: this only proves
   *serial API visibility*, not GPU-live polling.
   R9700: before rerunning BAR, use `bar_write_canary-gfx1201 --device 2 --probe-hip-only` under a fresh GPU lease. This dispatches a tiny HIP kernel with no HSA allocation or CPU BAR writes. A PASS only checks ordinary HIP dispatch, not Gate 2. Previous canary exited 1 before its HIP launch returned; possible launch stall/fault or command execution limit, not a confirmed data mismatch.
3. **Gate 2b cross-process serial IPC:** Run the matching
   `bar_ipc_canary --device HIP_ORDINAL` binary by an absolute path.
   It launches a fresh **CPU-only** child, passes an HSA GPU-allocation
   handle over an AF_UNIX socket and has the child attach as a CPU
   agent. If either export/attach or GPU verification fails, STOP:
   simple GPU-local CPU access is not enough for production.
   **Current XTX evidence (2026-11-09):** direct CPU-agent IPC
   import fails HSA 4104. Same-GPU IPC attach works on both XTXs,
   but the imported hostBaseAddress is not a writable CPU VMA.
   `rad50-tester --test ipc-map` is metadata only;
   `--test ipc-gpu-cpu` refuses writes if the mapping is unsafe.
   The separate `--test vmem-cap` checks local VMM handle/DMA-BUF
   export, GPU/CPU access and host VMA without storing bytes.
   On XTX0, VMM CPU get_access fails HSA 4100. **None of these
   results passes Gate 2b.** Do not run Gate 3 on this evidence.

4. **Gate 3 transfer A/B:** Only after Gates 1/2/2b PASS, run
   `bar_transfer_ab --device HIP_ORDINAL --iterations 30`. Report
   per-size (4 KiB to 21 MiB) and CPU 1/2/4 thread p50/p95 of
   *transfer* and *fully verified* makespan. It spawns fresh threads
   per copy, so its CPU 2/4-thread figures include startup overhead
   and cannot stand in for a pre-existing worker pool.
5. **Gate 3b contention:** In a separate, controlled same-lease arm,
   run GPU 0's mapped-host read pressure simultaneously with the
   device-under-test. Re-use RAD40's existing `host-pull.hip`
   harness with its independent PID and exact owned-process cleanup.
   Keep workload, shape, CPU affinity and clocks fixed. Do not run
   this arm before the uncontended control. Repeat AB/BA and log p95.
6. **CPU wake upper/lower bound:** Only after the RAD49 lease is
   released, `cpu_handoff_bench --mode spin|yield|sleep1|condvar
   --samples 3000` may measure *CPU-to-CPU* wake latency and
   per-sample CPU time. A GPU cannot notify a condition variable,
   so **none of these numbers measures GPU-to-CPU wake**. The existing
   GPU-to-CPU large-lane `worker_lanes.h` polling should be traced
   in production rather than inferred.
7. **Live publish/GPU wait (Gate 4):** After previous gates pass, run the architecture-matched `bar_live_ready_canary --device HIP_ORDINAL` first. It arms a GPU kernel, waits for the GPU's armed flag, then writes CPU BAR VRAM and publishes a separate host-ready generation. This is a concurrency *observation*, not a formal memory-ordering guarantee. Then create a **new**
   concurrent producer/consumer harness: enqueue a GPU waiter
   *before* CPU write, reuse chunk-ready generations, verify every
   element, test wrong flag/late write/recycle, and establish a
   driver-supported cross-VRAM/host-flag memory-ordering proof.
   `bar_ipc_canary` alone is insufficient. No engine work until then.
8. **Gate 5 captured rows:** Run the exact CPU combine reference,
   overflow fallback, no-route chunks, consumer-rank alternation
   and TP2 once-only oracle against real captures with no GPU serving
   mutation. Then review RAD51 before any implementation.

## D. Data required before a production go decision

For the *already-chunked* baseline and candidate, measure the following
per `(ticket id, consumer rank, placement epoch, combine generation, chunk)`:

- Helper GPU published final `chunks.seq[k][c]` by owner; stage trace.
- CPU combine first notices ready; wait time, combine start/end, CPU
  write start/end, actual BAR write visibility completion.
- CPU ready word release; GPU poller first sees it and GPU first/last
  row consumed; tail completion and parent retirement.
- Transferred bytes, concurrency, CPU core and time, HIP stream
  synchronization/launch counts and GPU busy/idle.
- p50/p95/p99 critical-path makespan and first-chunk latency,
  decode and prefill separately, under no contention and GPU0
  concurrent host-read pressure. Capture GPU MES idle power changes.

**Avoid incorrect arithmetic:** the RAD42 baseline already overlaps
CPU combine on chunk c+1 with engine GPU pulling chunk c. Summing
`bytes/GBps` for an entire plane double-counts time that was
already hidden under that overlap. Use the RAD50 critical-path
model with measured producer-ready intervals and CPU combine times.
Report direct-BAR write vs staging-then-copy separately.

With the illustrative (NOT measured) timing assumptions and RAD40's
contended bandwidth numbers in the model, a staged CPU copy *loses*
about 16% of modelled makespan; direct output wins only with an
optimistically unmeasured zero extra CPU store penalty. This is why
CPU wake, store latency and cross-process ordering are the hard gates.

## E. Safety and disposition

- No CPU/GPU hardware usage while another agent owns the lease;
  do not bypass the lease with an indirect command launcher.
- Do not run a GPU tool merely because utilization is currently 0%.
- Do not force HSA's attachment on another GPU or assume P2P exists.
- Fail closed on a mismatch, timeout or unsupported allocation.
- Keep status updated in [PROGRESS.md](PROGRESS.md); RAD50 remains
  pending, RAD51 is a *future-only* integration plan.
- No production edits, push, PR or merge in the experiment branch.

### New Linux DMA-BUF cross-process Gate 2b evidence

Use the existing `rad50-tester` with `--test dmabuf-map`, `dmabuf-write` (4 KiB) or `dmabuf-full` (4 MiB), plus `--device N --lease-confirmed`. XTX0 and XTX1 passed the first serial write/verify runs under an exclusive GPU lease. Parent ROCr GPU-local VMM export plus SCM_RIGHTS, child CPU-only mmap with DMA_BUF_IOCTL_SYNC start/end, and parent GPU exact-word validation succeeded. Direct HSA CPU-agent IPC still fails, but is not the selected transport. Before marking Gate 2b complete, require many generations, reuse, helper restart/close, failed import, per-rank isolation and concurrent publication; 4 MiB serial correctness alone is not a production throughput result. Check current GPU lease and KFD PIDs before all further runs.

### Real FP16 combine serial readiness (Gate 3b; partial acceptance)

The approved single `rad50-tester` includes `--test combine-ab --device 0|1 --iterations 10..200 --mode direct|staged [--rows 1|16|64|256|768] --lease-confirmed`. It invokes production `combine_row_f16` against deterministic BF16 normal/mask/fallback/NaN patterns and bit-checks every FP16 word on the GPU. 100 repeated 768-row generations passed on both XTXs; direct write into GPU-local VRAM beats host-mapped GPU-pull by ~79–81 us serial p50. Still do **not** mark concurrent Gate 4 passed, skip DMA-BUF SYNC, discard a failed partial log, or infer an engine speedup. The next gate must establish a consumer kernel that starts *before* helper writes, bounded waiting on separately published generation/ready, deterministic per-slot reuse/ack, crash/fault handling and a reliable GPU-visible release/acquire mechanism. Check GPU processes and acquire/release an exclusive lease for each hardware test session.

### Gate 4 live-GPU ready publication — HARD STOP until corrected (2026-11-09)

Initial `rad50-tester --test dmabuf-live` passes 1/16/256 rows ×10 on both XTXs, but **fails twice at 768 rows XTX0** when the already-running GPU observes stale ready=0 while the CPU has published ready=868001. The first GPU kernel used 256 polling threads; a one-poller diagnostic also failed. Treat this as a **correctness blocker**, not mere latency or a retry target. Do not run the old mapped-host-control mode repeatedly or increase timeout to hide it. Full logs and exact failure in `GATE4_LIVE_RESULTS.md`.

New fixed candidate `--test dmabuf-live-fg` uses HIP extended fine-grained host registration (hipExtHostRegisterUncached) of the shared memfd. It is compiled and card-free tested, **not hardware verified**, and unsupported for Navi4X according to the installed ROCm header. On a verified-free host, obtain exclusive GPU lease; first run XTX0 1 row ×10, then XTX0 768 rows ×10. Stop immediately if HIP host registration fails, any ready timeout occurs, GPU sees a stale generation or any FP16 mismatch. Do not proceed to XTX1, long repetitions, production integration or a relaxed gate after a failure. Preserve raw stdout, stderr, shell exit, timeout and truncation status; `live_analyze.py --provisional` retains partial evidence. Release the acquired lease and confirm idle/no KFD PIDs. If fine-grained memory fails, propose a separate coherent parent-relay or GPU-local-ready protocol with explicit cost and correctness tests.
