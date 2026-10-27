---
id: RAD21
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-23T02:13:57.784115+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P2
---

# moe-remote: pipeline the helper round trip within a layer by splitting the prefill batch

## Description

In the working Flash-Next layout the engine (R9700) publishes each MoE layer's activations to the XTX helpers, waits, then reads their rows back (moe_mailbox_add). That wait is serial: during a 7306-token prefill about 0.65 s of the 2.36 s TTFT is exposed helper wait, and moe_mailbox_add is 51.7% of engine device time. Splitting each layer's token batch into chunks lets chunk k's helper round trip overlap with chunk k+1's publish and the engine's local work (attention/shared expert, leftover experts on the R9700).

## Steps

1. Harness first (owner rule): extend the helper replay sweep (r3_moe_worker --replay-sweep) to time a layer split into N chunks with overlapped publish/compute/read, using captured requests, no model load.
2. Find the engine-side seam: where the PF/MoE layer publishes to the mailbox and waits (pinned radiance source + rad0002 hook), and whether the mailbox protocol can carry several in-flight chunks per layer (sequence per chunk).
3. Derive the chunk count from rows, link bandwidth (R9700 x4 upstream, ~3.8 GB/s hipMemcpy, no P2P) and measured helper time per row; no hand-set constant (derive-over-configure rule).
4. Implement in the moe-remote component and the rad0002 patch (bug-fix-in-patch rule if rad0002 changes).
5. --profile-ops A/B to show the exposed wait shrink, then end-to-end A/B in helper-ab.json vs the parent build.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

engines/radiance/components/moe-remote/ (r3_moe_worker.cpp, moe_remote_gpu.h, moe_lane_common.h)
engines/radiance/patches/rad0002_*/patch.py (engine-side publish/wait)
tools/lab/radiance/ replay harness config
docs/reference/engines/radiance/models/qwen3.8-flash-next.md

## Validation

Replay harness: overlapped chunks beat serial for the measured row sizes. Output bit-identical to unchunked (same per-row math). helper-ab.json 4 pairs: pp1024/pp4096 up, tg not worse. Exposed helper wait in --profile-ops falls.

## Effort & Risk

L / advanced. Risk: mailbox protocol change (version bump), more sync points, small chunks lose GEMM efficiency on the helpers. Build after RAD20 so the new waits are blocking from the start.

## Standards



## Acceptance Criteria

Exposed helper wait per prefill measurably lower; pp4096 end-to-end gain with complete separation vs the parent build; tg not worse; output bit-identical; chunk count derived, not configured.

## Notes

Next prefill lever after RAD18 (combine). Measurements from RAD18 notes. Ask GPT for design review and prior art (e.g. expert-parallel overlap in DeepSpeed-MoE / Megatron / vLLM / SGLang two-batch overlap).

2026-10-27, DeepEP/MORI-EP adaptation brief (verified against the code): kLaneSlots = 4 is a metadata ring only; activation and result planes are per rank (moe_remote_shm.h activation_offset[kMaxRanks], result_offset[kMaxRanks*kMaxRanks]). Several chunks in flight therefore need explicitly owned bulk slots with credits (FREE -> STAGING -> PUBLISHED -> HELPER_RUNNING -> HELPER_DONE -> MERGING -> RETIRED), released only after the last consumer has read them; the helper's done sequence alone does not prove the engine consumed a result. Order: (C1) replay cost model on captured requests and real helper timings, chunk counts derived from rows and route distribution; (C2) standalone multi-slot mailbox probe (generation, wrap, stop/fault, SIGKILL); (C3) trace the rad0002 after_route/merge seam (patch.py ~350-392) before choosing where to chunk; production only if C1 and C2 win. No extra helper GEMM streams (two-stream helper hung, RAD02); no per-layer CPU round trip; no resident GPU waits. Liveness (RAD24) must be merged first: the chunk protocol builds on helper_fault/notify.

## Execution gate review (2026-10-23)

- First PR is **harness + isolated scheduling experiment only**. Model per-chunk publish, helper execution, local engine work, engine read/merge, and x4 upstream bus contention using captured M/route distributions. Measure end-to-end *critical path*, not helper idle time in isolation.
- Gate engine patch work on an actual legal execution seam: row partition must preserve routing, expert token IDs, residual order, masking, per-row activation/outputs, sequence ownership, MTP state and CUDA/HIP event lifetimes. A layer's next chunk cannot read results before the matching generation is committed; stop/fault must retire all in-flight chunks.
- Use an empirical chunk estimator with a bounded search/adaptation; preserve a single-chunk path for small decode and unsafe shapes. No fixed model- or device-specific chunk constant. Exit early if x4 PCIe copies serialize the purported overlap or helper GEMMs lose too much efficiency.
- Follow-on production PR requires paired pp1024/4096, tg128/512/2048, KLD/top-1, stable fixed-text latency, exposed wait profile, and no increase in host traffic beyond the measured budget. Any numeric reduction-order change is gated by KLD, not assumed bit-identical.

## Change Log

- 2026-10-23T02:13:57.784115+00:00 (created-by): Created by agent
- 2026-10-23T02:14:14.836645+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:effort_risk, section:notes
- 2026-10-23T02:15:13.650842+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-10-27T02:35:22.779320+00:00 (updated-by): Updated: section:notes
