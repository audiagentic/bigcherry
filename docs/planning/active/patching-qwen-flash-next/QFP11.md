---
id: QFP11
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T16:33:36.231209+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Cut per-AllReduce boundary cost in the MTP verify step (~67 us idle x ~96 per step): AR inside one graph, fewer splits

## Description

Timed synctrace + rocprof census on production profile v2 (~80K): the host waits ~11.0 ms per MTP step for the target verify (llama_context::synchronize from server decode, 261 calls = 2874 ms of 3862 ms blocked); the draft sampling wait is only ~1.8 ms/step. Inside the verify the three tensor-split GPUs run kernels only ~42% of the time, so ~6.4 ms per step is idle, spread over ~96 AllReduce points = ~67 us of dead time per AR boundary, vs ~14 us for the cpu-root AR itself in isolation. Under -sm tensor the meta backend splits the graph at every AllReduce (each split is a separately cached/launched CUDA graph: ~195 cached instances per GPU, 2 MiB each, QFP06), so each boundary pays split launch/handoff + cross-GPU arrival skew + the host-root round trip.

## Steps

1. Decompose the 67 us: per-boundary timeline (rocprof kernel trace: last kernel of split N on each GPU -> AR kernel start/end -> first kernel of split N+1) on profile v2. 2. Prototype AR inside a single captured graph per GPU (device-side wait on the cpu-root epoch already exists in 1291; the split is a scheduler artefact) so one graph launch covers the whole verify step. 3. If (2) is infeasible, batch adjacent splits / reduce split count. 4. Measure ms/step ABBA and cached-graph count/memory.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Fewer graph launches per step (rocprof), GPU busy share up from ~42%, ms/step ABBA on profile v2 at 10K/80K, greedy identical, graph memory down.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Evidence runs: flashnext-v2-synctrace-timed (sync-tracer.c now records blocked time), flashnext-v2-profile (rank-census.py), flashnext-v2-1304-memlog. Related: QFP01 (1291 cpu-root AR), QFP06 (graph memory), QFP09 (rank skew), RNX11 (AR graph-replay hardening), PRBE66/67 (graph keying).

## Change Log

- 2026-10-03T16:33:36.231209+00:00 (created-by): Created by agent
