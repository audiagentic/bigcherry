---
id: BCOP40
order: 40
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-06T12:12:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# Disposition post-1326 scheduler host-input staging

## Description

Optimisation-audit disposition for QFP16 after 1326 reached the production enhancement set. QFP16/1326 remain the sole technical owner.

## What changed/discovered

- Subsequent BigCherry work already solved the dominant decode-side QFP16 bottleneck: safe staged 1326 measured +9.4% decode at ~8K and +7.8% at ~64K with greedy identity.
- The earlier QFP16 outer-HIP-graph proposal is no longer first priority; 1320 bounded that opportunity while 1325/1326 found and removed the larger scheduler-copy term.
- HIP's documented contract says non-pinned host memory makes hipMemcpyAsync synchronous. 1326's pageable scheduler staging is therefore a safe synchronization-elision path, not true H2D overlap.
- Current llama.cpp master still has the host-source synchronous scheduler fallback; upstream has not superseded 1326.
- vLLM provides the relevant transferable mechanism: pinned host staging plus completion-event-scoped reuse for genuinely non-blocking copies. This is a lifetime pattern, not a CUDA performance claim.

## Already acted upon

- 1319/1320/1325 isolated the submit path.
- 1326 owns and ships the <=4 MiB safe path.
- Unsafe caller-owned pinned-source prefill results are retained only as motivation, not promotable evidence.

## Unresolved action

First measure the residual large/prefill split-input cost on the current production stack at ubatch 512/2048/4096. If it is >=1 ms or >=3% of prefill wall time, extend 1326 with a bounded scheduler-owned pinned staging ring protected by completion events from every consuming Meta destination. Do not create another scheduler-copy owner or option.

## Terminal disposition

- **Stop/close:** residual <1 ms or <3% of prefill wall time.
- **Promote prototype:** discriminator passes, then pinned ring gives >=5% median prefill gain at ubatch 2048/4096, <=2% decode regression, greedy identity, multi-request/multi-ubatch slot-reuse correctness, and bounded pinned memory.
- **Reject prototype:** performance gate or any lifetime/correctness gate fails.
- **Outer HIP graph:** remain deprioritized unless fresh post-1326 profiling shows >=1 ms/round critical-path residual.

## Dependencies / references

- QFP16
- 1326_sched_async_host_inputs
- ROCm HIP host-memory and asynchronous-copy documentation
- current llama.cpp ggml_backend_sched_compute_splits
- vLLM pinned/event-scoped staging precedent
