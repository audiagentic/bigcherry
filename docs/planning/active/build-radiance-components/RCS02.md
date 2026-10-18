---
id: RCS02
order: 2
plan: build-radiance-components
state: pending
created-at: '2026-10-18T12:57:34.015384+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Split moe_remote_gpu.h (2,048 lines) into lane units - pure move

## Description

Break the lane code into single-purpose files with no behaviour change: lane_front (status/stats, activation copy, expert-table selection, routing prologue), lane_small (per-route GEMV), lane_large (grouped moe_gemm_q, long-run split), lane_io (VECIO helpers, activation prepare/reduce), lane_publish (done word, trace stamps), enqueue (enqueue_lane and lane dispatch). Target: no file over about 600 lines.

## Steps

1. Move code only; no edits to logic, names or flags.
2. Keep one include order and one translation unit per lane where the build allows.
3. After each file move: build the helper, run r3_moe_shm_test, then the replay sweep over the existing captures and require bitwise identity with the pre-move build.
4. Update TOOL_DISPOSITION and any docs that cite line ranges.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Replay sweep (captures/ and captures-small/) is bitwise identical before and after for every env variant; r3_moe_shm_test passes; patch-lint clean; one decode and one prefill benchmark within noise.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Depends on RCS01. The replay harness is the safety net, which is why this split is cheap.

## Change Log

- 2026-10-18T12:57:34.015384+00:00 (created-by): Created by agent
