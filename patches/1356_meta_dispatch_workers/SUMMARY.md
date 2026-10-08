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
