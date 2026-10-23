---
id: RAD19
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-23T00:48:25.505935+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: S
---

# moe-remote helper: derive expert slice split and dtype from the model, not Flash-Next constants

## Description

The moe-remote helper's tensor-parallel slice mode (R3_MOE_WORKER_ENGINE_RANKS=2, each helper computes its engine rank's slice of every expert) is hand-rolled in engines/radiance/components/moe-remote/moe_lane_common.h: kExpertWidth = 640, kRankWidth = 384 (rank 0 takes 384 columns, rank 1 the other 256, alternating by gate/up parity via rank_slice_lo/rank_slice_width), and kDtype = "w4nl64a8h". It mirrors radiance's own TP=2 split for Qwen3.8-Flash-Next but is our copy, not derived from the engine or the container, and moe_remote_gpu.h:92 refuses any expert width other than 640 in that mode. This violates the owner rule 'nothing model-specific in code; derive over configure' (CLAUDE.local.md, RCS09/RCS10). The current best layout uses ENGINE_RANKS=1 (full-width experts), so this only bites if the sliced mode is revived or another model is served.

## Steps

1. Find where radiance itself decides the per-rank expert split for --tp 2 (placement / weight sharding code in the pinned source) and whether it is exported (container metadata, plugin ABI, or the engine's published layout).
2. Make the helper take the split and expert dtype from that source (or from the container's shape: intermediate width, rank count) instead of kExpertWidth/kRankWidth/kDtype.
3. Remove the width-640 refusal in moe_remote_gpu.h once the split is derived; keep a refusal only for shapes the kernels genuinely cannot serve, with the reason stated.
4. Re-verify bit-identity of the sliced mode against the current constants on Flash-Next (replay harness, --compare), and with a second MoE model if one is available.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

engines/radiance/components/moe-remote/moe_lane_common.h (kExpertWidth, kRankWidth, kDtype, rank_slice_lo/width)
engines/radiance/components/moe-remote/moe_remote_gpu.h (width-640 refusal, ~line 92)
engines/radiance/components/moe-remote/r3_moe_worker.cpp (slice selection)
engines/radiance/components/moe-remote/tests (r3_moe_shm_test, r3_moe_backend_test)

## Validation

r3_moe_worker --compare / --replay-sweep on Flash-Next captures: sliced-mode output bit-identical before and after; component tests (r3_moe_shm_test, r3_moe_backend_test) pass; no remaining 640/384/w4nl64a8h literals in the helper's selection path.

## Effort & Risk



## Standards



## Acceptance Criteria

No Flash-Next constants (640/384/w4nl64a8h) left in the helper's selection path; split and dtype come from the container or the engine; sliced mode bit-identical to today on Flash-Next; a shape the kernels cannot serve is refused with the reason.

## Notes

Raised 2026-10-23 while discussing why the helper exists instead of using the engine as an expert host (RAD02 rationale: radiance has no expert-parallel mode; TP splits every layer and needs every rank on one kernel library). Not urgent: the working layout runs full-width experts. Related: RAD02 (inverted expert-parallel layout), RAD18 (helper transport work, R3_MOE_WORKER_COMBINE).

## Change Log

- 2026-10-23T00:48:25.505935+00:00 (created-by): Created by agent
