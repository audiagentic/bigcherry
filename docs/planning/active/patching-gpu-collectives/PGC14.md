---
id: PGC14
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-10-01T11:50:01.913289+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
---

# RCCL source patches: hostcall-free gfx1030 kernels and SHM transport throughput

## Description

RCCL source is on Brutus (`~/rccl-heterogeneous-src/rccl`, ROCm/rccl `57e58688`). Two problems:

1. chipset-attached RX 6900 XT/gfx1030 has no PCIe atomics; RCCL collective kernels request a hostcall buffer and fail at first launch (`Pcie atomics not enabled, hostcall not supported` / AQL dispatch failed). This is independent of protocol/SHM/P2P/MSCCL in the recorded tests.
2. with P2P disabled, large 21-42 MiB collectives on dual XTX reach only ~1.1 GB/s versus much higher measured host-DMA ceilings; prefill is collective dominated.

**Source correction, 2026-10-05:** RCCL `57e58688` already implements an asynchronous `NCCL_STEPS` copy-engine ring in `src/transport/shm.cc` when SHM cudaMemcpy mode is enabled. `shmSendProxyProgress()` and `shmRecvProxyProgress()` keep up to `NCCL_STEPS` transfers outstanding, use `cudaMemcpyAsync`, record per-slot events and publish FIFO/tail progress only after `cudaEventQuery` succeeds. Therefore do **not** add a parallel BigCherry ring. The transport work is to establish which SHM mode production actually takes, instrument the existing ring, then fix the measured limiter: mode selection, step/chunk geometry, one-direction-only CE, proxy starvation/polling, copy granularity, or insufficient inflight depth.

This correction supersedes the earlier conceptual `bc_shm_slot` design in this plan.

**2026-10-10 SHM CE rebaseline:** Independent commit `9ded4c6e84b1` (2026-10-02) recorded `NCCL_SHM_USE_CUDA_MEMCPY=1` hanging `llama-bench`. Its timeout did not propagate to the sweep exit code. This is a historical failed-run report, not a current-pin reproduction or root-cause proof. **Do not promote CE or tune ring depth until a bounded single-arm test proves no hang, correct data and actual RCCL SHM selection.** The gfx1030 hostcall lane is independent.

## Steps

1. Hostcall lane: inspect code objects for `hidden_hostcall_buffer`; identify device `printf/assert/abort` users; build a no-hostcall gfx1030 variant and verify with rccl-tests before combining transport changes.
2. Establish exact SHM baseline matrix at 1/4/10/21/42 MiB: direct default; `SHM_USE_CUDA_MEMCPY=1` with memcpy mode `{send=1,recv=2,both=3}`; locality `{send=1,recv=2}`; protocol/channel/buffer controls. Record which proxy progress functions are actually installed.
3. Instrument existing `shmProxyInfo`/progress loops: submitted/completed bytes, D2H/H2D copies, inflight steps (`transmitted-done`), max inflight, event-not-ready polls, data-not-ready polls, window-full polls, copy size histogram, per-copy completion latency, proxy idle cycles.
4. Use the instrumentation to classify the bottleneck before changing code:
   - `max_inflight ~= 1` and many source-not-ready polls -> producer/device protocol is not feeding the CE ring;
   - high inflight + low CE bandwidth -> copy size/stream/topology issue;
   - many window-full polls -> deepen steps or advance completion faster;
   - tiny copy histogram -> larger buffer/step or safe adjacent-step coalescing;
   - send CE fast but receive direct slow (or reverse) -> mode=3/bidirectional CE candidate;
   - proxy idle/starvation -> proxy scheduling/progress issue, not FIFO geometry.
5. First code experiment: no algorithm change. Add runtime diagnostics + explicit environment screen for the existing CE path. Promote configuration-only if it closes most of the gap.
6. Only if measured: tune `NCCL_STEPS`/step size or `sliceSteps/chunkSteps` relationship. Keep FIFO sequence semantics unchanged.
7. Only if copies are too small and adjacent ready slots are common, prototype contiguous ready-step coalescing in the proxy: one larger async memcpy plus one completion record for a run, without publishing any tail before the whole run completes.
8. Only if a single CE stream itself is limiting after size/depth tuning, test per-direction/multiple streams; do not assume more streams improve PCIe DMA.
9. Re-run rccl-tests and BigCherry pp1024/2048/4096. Compose independently with PGC15 token-tiled overlap.
10. Ship as pinned BigCherry-managed RCCL build only after stock-vs-patched correctness and topology gates.

## Detailed Solution & Technical Design

### Existing pipeline: preserve it

At `57e58688`, `shmProxyInfo` already owns:

```cpp
struct shmProxyInfo {
  struct ncclRecvMem* ceRecvMem;
  char* devFifo;
  char* shmFifo;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  uint64_t step;
  cudaStream_t stream;
  cudaEvent_t events[NCCL_STEPS];
  ncclShmIpcDesc_t desc;
};
```

Connect allocates the device FIFO, pinned receive metadata, nonblocking stream and one event per step:

```cpp
NCCLCHECKGOTO(ncclCudaCalloc(&proxyInfo->devFifo,
                             proxyState->buffSizes[NCCL_PROTO_SIMPLE]), ret, fail);
NCCLCHECKGOTO(ncclCudaHostCalloc(&proxyInfo->ceRecvMem, 1), ret, fail);
CUDACHECKGOTO(cudaStreamCreateWithFlags(&proxyInfo->stream,
                                        cudaStreamNonBlocking), ret, fail);
for (int i = 0; i < NCCL_STEPS; i++) {
  CUDACHECKGOTO(cudaEventCreate(proxyInfo->events+i), ret, fail);
}
```

Do not replace this with another slot/generation abstraction unless a proven correctness bug requires it.

### Existing send/D2H progress loop

Current semantics are already pipelined:

```cpp
if (sub->transmitted < sub->done + NCCL_STEPS &&
    sub->transmitted < sub->nsteps) {
  int buffSlot = (sub->base + sub->transmitted) % NCCL_STEPS;
  volatile struct ncclConnFifo* connFifo = resources->ceRecvMem->connFifo;
  volatile uint64_t* recvTail = &resources->ceRecvMem->tail;

  if (*recvTail > sub->base + sub->transmitted) {
    int size = connFifo[buffSlot].size;
    CUDACHECK(cudaMemcpyAsync(
        resources->shmFifo + buffSlot*stepSize,
        resources->devFifo + buffSlot*stepSize,
        size, cudaMemcpyDeviceToHost, resources->stream));
    CUDACHECK(cudaEventRecord(resources->events[buffSlot], resources->stream));
    resources->recvMem->connFifo[buffSlot].size = size;
    __sync_synchronize();
    sub->transmitted += args->sliceSteps;
  }
}
```

Completion then queries the event and only afterwards publishes SHM tail:

```cpp
if (sub->done < sub->transmitted) {
  int buffSlot = (sub->base + sub->done) % NCCL_STEPS;
  cudaError_t res = cudaEventQuery(resources->events[buffSlot]);
  if (res != cudaErrorNotReady) CUDACHECK(res);
  if (res == cudaSuccess) {
    sub->done += args->sliceSteps;
    resources->recvMem->tail = sub->base + sub->done;
  }
}
```

Receive/H2D is symmetric: it waits for SHM tail, copies `shmFifo -> devFifo`, records the slot event, then publishes `ceRecvMem->tail` only after event completion.

### Mode selection is the first suspect, not ring depth

`initCeOperation()` currently does:

```cpp
useMemcpySend = ncclParamShmUseCudaMemcpy() &&
                (ncclParamShmMemcpyMode() & 1);
useMemcpyRecv = ncclParamShmUseCudaMemcpy() &&
                (ncclParamShmMemcpyMode() & 2);

if (useMemcpySend) {
  shmTransport.send.proxyConnect  = shmSendProxyConnect;
  shmTransport.send.proxyProgress = shmSendProxyProgress;
}
if (useMemcpyRecv) {
  shmTransport.recv.proxyConnect  = shmRecvProxyConnect;
  shmTransport.recv.proxyProgress = shmRecvProxyProgress;
}
```

and `SHM_USE_CUDA_MEMCPY` defaults to `0`; `SHM_MEMCPY_MODE` defaults to sender-side only. Therefore first establish whether the ~1.1 GB/s baseline is direct SHM rather than this CE path. Do not modify `NCCL_STEPS` before this is known.

Add once-only diagnostics at init:

```cpp
INFO(NCCL_INIT|NCCL_SHM,
     "BigCherry SHM diag: ce=%d mode=%d send=%d recv=%d locality=%d simpleBuff=%zu steps=%d",
     ncclParamShmUseCudaMemcpy(), ncclParamShmMemcpyMode(),
     useMemcpySend, useMemcpyRecv, shmLocality,
     proxyState_or_comm_simple_buffer_size, NCCL_STEPS);
```

Use the correct available object for buffer size at the call site; do not introduce a global just for logging.

### Instrumentation fields

Extend `shmProxyInfo` behind a compile/runtime diagnostic gate:

```cpp
struct bcShmDiag {
  uint64_t submittedBytes;
  uint64_t completedBytes;
  uint64_t submittedCopies;
  uint64_t completedCopies;
  uint64_t dataNotReadyPolls;
  uint64_t eventNotReadyPolls;
  uint64_t windowFullPolls;
  uint32_t maxInflight;
  uint32_t minCopy;
  uint32_t maxCopy;
};

struct shmProxyInfo {
  // existing fields...
#ifdef BIGCHERRY_RCCL_DIAG
  struct bcShmDiag bc;
#endif
};
```

At submission:

```cpp
#ifdef BIGCHERRY_RCCL_DIAG
resources->bc.submittedBytes += size;
resources->bc.submittedCopies++;
resources->bc.minCopy = resources->bc.minCopy == 0 ? size : std::min(resources->bc.minCopy, (uint32_t) size);
resources->bc.maxCopy = std::max(resources->bc.maxCopy, (uint32_t) size);
resources->bc.maxInflight = std::max(resources->bc.maxInflight,
    (uint32_t) (sub->transmitted + args->sliceSteps - sub->done));
#endif
```

At readiness/window branches, count why progress could not occur. Keep diagnostics host-only; no device printf (hostcall problem).

### Deferred coalescing: correct slot extent before implementation

**Withdraw the previous coalescing pseudocode.** It admitted a short first FIFO slot followed by a full slot and used `sum(size)` for the contiguous memcpy length. With `stepSize=256` and sizes `[64,256]`, it copies 320 bytes although the second slot ends at offset 512. The proposal also assumed `args->sliceSteps==1`, which is not guaranteed. This is a deterministic host-level counterexample to the proposed sketch, **not** a proven defect in stock RCCL (which does not coalesce).

Only reconsider after CE hang-free correctness and traces showing small copies. A first prototype must require `args->sliceSteps==1`, no ring wrap, producer tail proving every slot ready, all non-final slots exactly `stepSize`, final size in `(0,stepSize]`, and memcpy extent `(run_slots-1)*stepSize+last_size`. Publish every FIFO slot size; publish the batch tail only after its completion event. Reject short non-final slots, sliceSteps>1, wrap, missing metadata and event reuse to the existing single-step path. Verify monotone tails, buffer generation and consumer visibility with a host fixture before hardware.

### Do not conflate SHM transport with CPU-root reduction

RCCL SHM proxy transports protocol buffers between GPUs/host-visible SHM; it is not BigCherry's CPU f32 sum worker. Prior CPU-root large-message failures do not prove the existing RCCL CE proxy is doomed. Conversely, faster standalone H2D/D2H does not prove the collective protocol can consume/produce steps fast enough. Instrument both readiness and DMA.

### Interaction with PGC15

PGC14 reduces service time of each RCCL collective. PGC15 changes when/range size collectives are issued. Keep them independently selectable and run four arms where possible: stock/patched RCCL × whole/tiled AR.

### CE admission, observability and terminal gate (2026-10-10)

- **Pinned source:** `ROCm/rccl@57e58688/src/include/param.h::NCCL_PARAM` prefixes `NCCL_` and caches on first read; `src/transport/shm.cc::initCeOperation` (lines ~501-520) installs process-global callbacks once. Correct keys: `NCCL_SHM_USE_CUDA_MEMCPY=1`, `NCCL_SHM_MEMCPY_MODE=1|2|3`, `NCCL_SHM_LOCALITY=1|2`. Use a **fresh process per arm**, never an in-process environment toggle.
- `shmCanConnect` may reject SHM, and `shmSendProxyProgress`/`shmRecvProxyProgress` use CE only for `NCCL_PROTO_SIMPLE`. Force the **RCCL** provider on tensor-split prefill, not BigCherry's adaptive/internal two-GPU host AllReduce. Record the selected provider, SHM transport, protocol and callback before timing. Use direct-SHM stock control, then send-only mode 1, recv-only mode 2, both mode 3, each with `NCCL_PROTO=Simple`, finite timeout, and `rccl-tests` correctness.
- `tools/lab/rccl/rccl-env-sweep.sh` formerly printed `SWEEP_DONE` after timeout/crash/empty CSV. Its fail-closed receipt now records `status.tsv` with per-arm benchmark and CSV exits; a failed arm produces `SWEEP_INCOMPLETE` and nonzero overall exit. A historical timeout is **never** a zero-throughput measurement or successful sweep.
- First diagnose whether a current-pin hang is sender/receiver/both, readiness/tail publication, event completion, protocol or SHM selection. Do not queue a run over an active Brutus GPU/host-exclusive lane. If CE hangs/corrupts, reject the mode and retain direct SHM. If all modes are correct but none materially improves matched RCCL-only collective wall and pp2048/4096, close CE tuning. Only then profile the existing `NCCL_STEPS` ring; no new ring, scheduler or allocator.
- Upstream ROCm/rccl PR #2187 (open 2026-10-10) identifies gfx12 LL memory-ordering deadlock risk. `NCCL_PROTO=Simple` is a correctness control for gfx1201, not a proven speedup. vLLM custom AR requires P2P and SGLang PCIe-IPC requires a distinct workspace/peer contract; neither is a drop-in no-P2P substitute.

**Cheap discriminators performed:** six deterministic host FIFO/env assertions, shell syntax, and a reduced fake-benchmark test: success exits 0/`SWEEP_DONE`, CE exit 124 yields `SWEEP_INCOMPLETE`/exit 1 and an explicit per-arm status. No real RCCL, HIP or GPU run.

## Code Samples & Guidance

Implementation sequence:

```text
A. no code: direct vs CE-send vs CE-recv vs CE-both × locality screen
B. add host-only proxy diagnostics to the existing progress loops
C. identify source-not-ready / window-full / CE-bandwidth / proxy-starvation limiter
D. tune existing buffer/steps/chunks first
E. coalesce adjacent steps only if small-copy evidence supports it
F. multiple CE streams only if one stream is demonstrably the limiter
G. model ABBA, then compose with PGC15
```

Do not introduce a second FIFO/ring in A-D.

## Files

- RCCL `src/transport/shm.cc` at `57e58688`: `shmProxyInfo`, `shmSendProxyConnect`, `shmRecvProxyConnect`, `shmSendProxyProgress`, `shmRecvProxyProgress`, `initCeOperation`.
- RCCL proxy/channel code only if diagnostics show scheduling starvation.
- Device protocol code only if source/tail publication is the limiter.
- BigCherry `tools/lab/rccl/`: mode matrix, rocprof timelines, rccl-tests wrapper, stats parser.
- Managed RCCL patch provenance/build recipe.

## Validation

**First gate:** process-isolated direct SHM and CE modes 1/2/3; exact `NCCL_` keys; confirmed RCCL/SHM/Simple admission; valid per-arm exit/CSV receipt; no hang or wrong result. Preserve the 2026-10-02 hang as historical, not current performance evidence. A failed arm invalidates the sweep.


For every source change: rccl-tests `#wrong=0`, sizes around step/chunk/ring boundaries, >=100 warmed repetitions, 2x XTX and XTX+R9700/3-rank supported topology. Record p50/p90 latency, algbw/busbw, D2H/H2D engine busy, actual copy-size distribution and max inflight.

BigCherry: pp1024/2048/4096 plus Flash-Next long fill; compare collective critical-path wall and prefill t/s. Decode control <=1% regression.

Hostcall lane separately: gfx1030 mixed pair/4-GPU launch, no `hidden_hostcall_buffer`, correctness and soak.

## Effort & Risk

L. Hostcall patch and transport patch are separable. Transport risk is lower when extending existing step/event state than adding another queue; remaining risks are FIFO/tail ordering, proxy starvation, copying unpublished padding during coalescing, and microbenchmark-only wins.

## Standards

Pinned RCCL provenance; existing FIFO/step state is authoritative; diagnose before redesign; no device printf/assert dependency on gfx1030; stock RCCL always selectable; correctness + in-model evidence required.

## Acceptance Criteria

- Every CE candidate passes actual SHM activation, correctness, bounded no-hang and per-arm receipt before bandwidth claims. Failed arms fail the sweep; direct SHM stays selectable.
- No coalescing until the short-first/sliceSteps/wrap host fixtures and event/tail-lifetime invariants pass.


- gfx1030 RCCL kernels launch without hostcall dependency and pass correctness.
- Exact reason for 1.1 GB/s baseline is recorded (direct/CE mode plus proxy diagnostics).
- No <=1 MiB regression >2%.
- 21-42 MiB host-staged transport reaches >=4 GB/s first-stage acceptance and targets >=6 GB/s on dual XTX, or the measured protocol ceiling is documented with a different owner for the limiter.
- Flash-Next prefill >=5% or collective critical wall >=20% improvement, decode <=1% regression.
- No duplicate host-stage ring is introduced unless existing `NCCL_STEPS` machinery is proven structurally incapable.

## Notes

2026-10-05 source audit correction: `57e58688/src/transport/shm.cc` already has an event-tracked `NCCL_STEPS` CE pipeline. Earlier plan language proposing a fresh fixed-depth ring was superseded. First priority is mode/limiter attribution and tuning of that implementation.

2026-10-06 RCCL ALGORITHM / PROTOCOL SCREEN - REJECTED (run chain3, queue-rccl-screen.sh b11402, production build b-fadef-b11402e, Flash-Next production with MTP, ~99K-token prefill (99,342 tokens), ABBA per setting, A = default RCCL, B = the setting). NCCL_PROTO=Simple: A 970.5 / 984.9, B 981.8 / 981.1 t/s - no change. NCCL_PROTO=LL: A 982.6 / 983.0, B 272.8 / 271.9 t/s - 3.6x slower. NCCL_PROTO=LL128: A 980.0 / 982.7, B 985.6 / 983.4 - no change. NCCL_ALGO=Tree: A 981.9 / 982.0, B 981.2 / 982.7 - no change (decode t/s 59.6 / 60.3 vs 57.5 is a different generated text and acceptance, 338/518 and 339/515 vs 332/535; time per decode step is the same 16.6 ms). NCCL_ALGO=Ring: A 981.6, B 984.3 / 982.4 with the last A arm still running when recorded - no change. No RCCL algorithm or protocol setting improves the tensor-split all-reduce on this host and one makes it much worse; the default is already the best of the set. This closes the env-only lever. What remains for the all-reduce cost is structural (PGC15: fewer or larger collectives, tiling), not configuration.

## Change Log

- 2026-10-10: SHM CE hang-first rebaseline; corrected fail-open sweep receipt, env/process/protocol gates and unsafe coalescing sketch. No new RCCL build or GPU benchmark.


- 2026-10-01T11:50:01.913289+00:00 (created-by): Created by agent.
- 2026-10-05: Expanded prefill transport scope.
- 2026-10-05: Re-grounded transport design against RCCL 57e58688; replaced duplicate-ring proposal with concrete existing SHM proxy diagnostics/tuning/coalescing code.
- 2026-10-06T03:37:19.080217+00:00 (updated-by): Updated: section:notes
