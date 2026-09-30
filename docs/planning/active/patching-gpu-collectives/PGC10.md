---
id: PGC10
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-30T05:20:19.155660+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# 3-GPU (2x XTX + R9700) AllReduce: adaptive root3/RCCL hybrid for models that need three cards

## Description

Goal (owner): run larger models / larger variants that need three cards (2x 7900 XTX gfx1100 + R9700 gfx1201, 80 GB VRAM) as fast as possible, combining RCCL with BigCherry's host-pipeline optimisations. Baseline measured on Qwen3.8-27B Q8_0 across 3 GPUs (ab-3g-allreduce, 6 balanced rounds): RCCL tg512 91.8 / pp4096 1098; 1244 root3 decode +2.0% (tg512) / +3.4% (tg2048) vs RCCL but prefill -68% (root3 has no large-message path, same as `none`). So the target is the same hybrid as dual XTX: root3/host for small (decode) reductions, RCCL for large (prefill).

## Steps

1. Extend 0840 adaptive to N=3: small -> 1244 root3 internal path, large -> RCCL (0840 currently assumes n_devices == 2 for its internal side).
2. Apply the decode-latency fixes from the Patch B review (1275: slot-sync host->stream, size-adaptive small-kernel geometry, later fused residual) to the root3 path as well as N=2.
3. Uneven split for mixed cards (R9700 32 GB vs XTX 24 GB, different bandwidth): sweep --tensor-split.
4. Target model: unsloth Qwen3.8-27B BF16 (~55 GB, needs three cards; /mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/BF16/); later a 60-75 GB MoE.
5. Balanced A/Bs: RCCL vs root3 vs adaptive(N=3), then with 1275 switches.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Balanced server A/B on the 3-GPU 27B BF16 config with CI95 excluding zero; prefill must not regress vs RCCL; decode must beat RCCL. Heterogeneous-arch RCCL safety via 1225 guard. vLLM (radiance-vllm) must be stopped for R9700 tests and restarted after.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

The 3-GPU Q8_0 27B runs slower than the dual-XTX host path (tg512 95.5), so Q8_0 27B stays on two cards; this item is for models that do not fit on two. 1244 forces F32 wire for N=3; 1272 codec extension to root3 is Patch C territory (gfx1201 fp8 sender).

## Change Log

- 2026-09-30T05:20:19.155660+00:00 (created-by): Created by agent
