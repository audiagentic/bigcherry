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

2026-09-30 design review with dev-gpt-agent (req_ba6aa3f60e0e4f92) agreed: decode AllReduce is fixed-latency bound (per small AR: one 8x256 mapped-host kernel per GPU for ~10 KiB bf16, ev.ker records, two host hipEventSynchronize in acquire_slot from AR #3; AR outside the device graph). Patch B (pipelined copy-engine wire) dropped. 1275_ar_small_latency: BIGCHERRY_AR_SLOT_SYNC=host|stream|none (none primary; proven safe for 2-slot ring incl. pinned-host reuse and root3 as long as collectives are not skipped/reordered and each AR is single-chunk), BIGCHERRY_AR_SMALL_BLOCKS {1,2,4,8} + _THREADS {128,256}, trace host_enqueue_us/slot_wait_us, marker with n_devices/blocks/threads/slot_sync; covers 1244 root3. 1276: 0840 adaptive composes with 1244 for N=3 (small->root3, large->RCCL) and BIGCHERRY_AR_ROOT3_ROOT=0|1|2 (default 0; choose root from measurements, not architecture). Authoring requested (req after ba6aa3f6).

2026-10-01 1275 A/B on the dual-XTX host path (27B Q8_0 MTP, 6 balanced rounds, ab-27b-ar-small): BIGCHERRY_AR_SLOT_SYNC=none vs pristine tg512 +0.04% (CI -0.16..+0.23), tg2048 -0.01%, pp4096 -0.24% -> neutral; BIGCHERRY_AR_SMALL_BLOCKS=1 vs pristine tg512 -8.56%, tg2048 -8.47%, pp4096 -2.39% -> strongly worse. MTP acceptance identical (82.22%). Conclusion: the fixed 8x256 small-AR grid is not oversized and host slot syncs are not the decode bottleneck; the design review's fixed-latency hypothesis for these two knobs is refuted. 1275 stays untested (no gain); next question is whether MORE than 8 blocks helps (needs the arrival-ring layout widened).

2026-10-04 status: realised for Flash-Next by patch 1291_ar_cpu_root (--allreduce cpu-root): CPU-root one-shot AR for <= 64 KiB f32 messages on 3 GPUs (2x XTX + R9700), RCCL above; +5-7% decode vs RCCL, prefill unchanged; the large-message host path lost to RCCL and is off. Detail and follow-ups in patching-qwen-flash-next QFP01. Remaining scope here: generalising the adaptive root/RCCL policy to other models.

## Change Log

- 2026-09-30T05:20:19.155660+00:00 (created-by): Created by agent
- 2026-09-30T06:29:06.022444+00:00 (updated-by): Updated: section:notes
- 2026-09-30T14:44:36.752908+00:00 (updated-by): Updated: section:notes
- 2026-10-03T15:21:51.445491+00:00 (updated-by): Updated: section:notes
