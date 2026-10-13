---
id: BCOP129
order: 129
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T03:10:35+00:00'
created-by: agent
priority: P2
work: S
---

# PRBE41: reconcile 1263 single-GPU admission and measured pp512 ceiling

## Discovery / disposition

Patch 1263 is implemented but `untested` at b11474. Its `model.split_mode() != LLAMA_SPLIT_MODE_TENSOR` selector **declines** channels-major under tensor split and emits a negative marker; prior TESTING.md incorrectly advertised `-sm tensor` as the positive lane. Metal layout rejection was omitted from the port. The AMD #52 roofline gain came chiefly from CONCAT (1166 -> 34 ms), not a claimed 221 ms CONT reduction. Existing BigCherry b11126/b11233 validation has 24 historical records: four validated gfx1100 b11233 pp512 sessions +1.13%..+1.43%, and the best gfx1201 CI95-low is +2.59%; none reaches the >=3% CI95-low promotion floor. No new GPU speedup claimed.

## Ownership / active-work fence

PRBE41/1263 is the sole channels-major SSM_CONV owner. PRBE114 owns future Meta split-state/reshape propagation. Active QFP49/1360/1361 scheduler, transfer stream and stagger-pair work is excluded and untouched, as are QFP50/1359, QFP17/1330 and QFP41/1356. Last substantive independent PRBE41 plan change 2026-10-08 06:47 UTC; 1263 moved mechanically 2026-10-09 04:20 UTC; no independent PRBE41/1263 changes, submissions or queues in the preceding 12 hours. Recent BCOP124-128 do not own this slice.

## Bounded next action / terminal outcome

Use existing 1263 producer/evidence. First prove current b11474 single-GPU exposed CONCAT/CONT cost and actual positive kernel marker; if <2.9126% E2E, retire further optimisation. If eligible, run current-pin direct op, state/MTP/graph and four-session paired pp512/1024/4096 controls; require >=3% CI95-low E2E and <=1% control regression. Tensor split is **negative-only** until PRBE114 separately resolves its Meta propagation, without racing QFP49. Reject unsupported Metal paths. Do not add a new scheduler, allocator, layout API or duplicate experiment queue. Full technical design belongs only to PRBE41.

## Evidence

`1263/patch.py`, `1263/validation/producer.py`, `1263/evidence/{validation.json,withdrawn.json}`, PRBE41/PRBE114, [AMD #52](https://github.com/AMD-Ecosystem/llama.cpp/pull/52), [upstream #30149](https://github.com/ggml-org/llama.cpp/pull/30149), [vLLM causal_conv1d](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py), [SGLang KDA](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kda_backend.py). Source-static checks 10/10; host layout/state-tail cases 144/144, zero mismatches. No repository pytest, HIP build, GPU benchmark or new hardware qualification in this run.
