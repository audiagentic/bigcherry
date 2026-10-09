# 1356_meta_dispatch_workers

**Status:** untested
**Plan item:** QFP41

## Mechanism

`ggml_backend_meta_graph_compute` currently submits each simple backend's `cgraph_main` serially from one host
thread. With `BIGCHERRY_META_DISPATCH_THREADS=1`, this package creates one persistent worker per simple backend
and submits the same subgraph index to those workers concurrently. The caller joins all submissions before any
AllReduce/collective decision, exactly where the serial loop had completed all host submissions.

No thread is created per graph. Default is off during qualification.

## Shared-state boundary

Caller-thread only:
- meta graph rebuild and `backend_configs[*].cgraphs/nodes` mutation;
- split-state cache and rotating simple-tensor containers;
- 1340 arena planning/binding and graph allocation;
- auxiliary AllReduce graphs, copy scheduling, collective selection/fallback and comm context;
- scheduler callbacks/input staging before/after meta graph compute.

Worker-owned during a job:
- exactly one distinct `backend_config.backend`;
- its simple-backend graph submission and backend-local CUDA/HIP graph/pool state.

The join occurs before the caller reads the submitted subgraphs for AllReduce. Workers are destroyed before the
simple backends in the meta-context destructor.

## Evidence boundary

QFP41 records ~40% split idle as the opportunity, not a proven dispatch-thread saving. No performance or thread-safety
claim is made until identity, repeatability, stress and hardware ABBA pass.

## Shared state: CUDA graph capture (2026-10-09)

First hardware run (b-metamem-mdw1, Flash-Next production profile, two ABBAs per depth): prefill +2.5% at 8K, +0.5 to
+1.5% at 24K, +0.3% at 98K with every threaded run above every baseline run, but 5 of 12 threaded runs produced a
different greedy text (one garbage). Bisect on the same binary at 8K, three ABBAs per variant: workers only 2 of 6
different; fusion off 5 of 6 different; `GGML_CUDA_DISABLE_GRAPHS=1` 0 of 6 different with the prefill gain intact
(1249-1289 against 1209-1238 t/s).

Cause: `ggml_backend_cuda_graph_compute` captures, instantiates and updates HIP graphs per device, and with the
workers on those run concurrently with another device's capture or launch. Fix in this package
(`ggml/src/ggml-cuda/ggml-cuda.cu`): a process-wide `std::shared_mutex`, taken exclusively by a capturing call and
shared by a replaying call, held until the call returns. It is only taken when `BIGCHERRY_META_DISPATCH_THREADS` is
on, so the default path is unchanged. Direct (non-graph) evaluation stays unlocked; the graphs-off variant showed it
is safe across devices.

Re-test required after this change: repeated-run identity at 8K (at least six threaded runs), ABBA at 8K / 24K / 98K,
and a long decode.
