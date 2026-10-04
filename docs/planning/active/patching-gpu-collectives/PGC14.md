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

### Optional contiguous-step coalescing: only after evidence

If traces show many small consecutive ready slots, combine a non-wrapping run. Preserve FIFO sizes and publish completion only after the whole memcpy finishes.

Add per-slot batch metadata:

```cpp
struct shmProxyInfo {
  // existing...
  uint8_t batchSteps[NCCL_STEPS]; // 0 unless this slot is head of a submitted run
};
```

Submission sketch:

```cpp
const int first = (sub->base + sub->transmitted) % NCCL_STEPS;
int runSteps = 0;
size_t runBytes = 0;

while (runSteps < bcMaxCoalesceSteps &&
       sub->transmitted + runSteps*args->sliceSteps < sub->nsteps &&
       sub->transmitted + runSteps*args->sliceSteps < sub->done + NCCL_STEPS) {
  const int slot = (first + runSteps*args->sliceSteps) % NCCL_STEPS;
  if (slot < first) break; // ring wrap: keep one contiguous pointer range
  if (*recvTail <= sub->base + sub->transmitted + runSteps*args->sliceSteps) break;

  const int sz = connFifo[slot].size;
  // Require exact contiguous occupancy; if variable-size holes exist, stop.
  if (runSteps > 0 && sz != stepSize) break;
  runBytes += sz;
  runSteps++;
}

if (runSteps > 0) {
  CUDACHECK(cudaMemcpyAsync(resources->shmFifo + first*stepSize,
                            resources->devFifo + first*stepSize,
                            runBytes, cudaMemcpyDeviceToHost, resources->stream));
  CUDACHECK(cudaEventRecord(resources->events[first], resources->stream));
  resources->batchSteps[first] = runSteps;
  sub->transmitted += runSteps * args->sliceSteps;
}
```

Completion uses `batchSteps[first]` to advance `done` and publish tail for the whole run after the event succeeds. This code is intentionally conservative: if current FIFO layout/size semantics do not guarantee contiguous bytes for a run, reject coalescing rather than copying padding/unpublished data.

### Do not conflate SHM transport with CPU-root reduction

RCCL SHM proxy transports protocol buffers between GPUs/host-visible SHM; it is not BigCherry's CPU f32 sum worker. Prior CPU-root large-message failures do not prove the existing RCCL CE proxy is doomed. Conversely, faster standalone H2D/D2H does not prove the collective protocol can consume/produce steps fast enough. Instrument both readiness and DMA.

### Interaction with PGC15

PGC14 reduces service time of each RCCL collective. PGC15 changes when/range size collectives are issued. Keep them independently selectable and run four arms where possible: stock/patched RCCL × whole/tiled AR.

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

For every source change: rccl-tests `#wrong=0`, sizes around step/chunk/ring boundaries, >=100 warmed repetitions, 2x XTX and XTX+R9700/3-rank supported topology. Record p50/p90 latency, algbw/busbw, D2H/H2D engine busy, actual copy-size distribution and max inflight.

BigCherry: pp1024/2048/4096 plus Flash-Next long fill; compare collective critical-path wall and prefill t/s. Decode control <=1% regression.

Hostcall lane separately: gfx1030 mixed pair/4-GPU launch, no `hidden_hostcall_buffer`, correctness and soak.

## Effort & Risk

L. Hostcall patch and transport patch are separable. Transport risk is lower when extending existing step/event state than adding another queue; remaining risks are FIFO/tail ordering, proxy starvation, copying unpublished padding during coalescing, and microbenchmark-only wins.

## Standards

Pinned RCCL provenance; existing FIFO/step state is authoritative; diagnose before redesign; no device printf/assert dependency on gfx1030; stock RCCL always selectable; correctness + in-model evidence required.

## Acceptance Criteria

- gfx1030 RCCL kernels launch without hostcall dependency and pass correctness.
- Exact reason for 1.1 GB/s baseline is recorded (direct/CE mode plus proxy diagnostics).
- No <=1 MiB regression >2%.
- 21-42 MiB host-staged transport reaches >=4 GB/s first-stage acceptance and targets >=6 GB/s on dual XTX, or the measured protocol ceiling is documented with a different owner for the limiter.
- Flash-Next prefill >=5% or collective critical wall >=20% improvement, decode <=1% regression.
- No duplicate host-stage ring is introduced unless existing `NCCL_STEPS` machinery is proven structurally incapable.

## Notes

2026-10-05 source audit correction: `57e58688/src/transport/shm.cc` already has an event-tracked `NCCL_STEPS` CE pipeline. Earlier plan language proposing a fresh fixed-depth ring was superseded. First priority is mode/limiter attribution and tuning of that implementation.

## Change Log

- 2026-10-01T11:50:01.913289+00:00 (created-by): Created by agent.
- 2026-10-05: Expanded prefill transport scope.
- 2026-10-05: Re-grounded transport design against RCCL 57e58688; replaced duplicate-ring proposal with concrete existing SHM proxy diagnostics/tuning/coalescing code.
