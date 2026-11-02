---
id: RAD45
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-02T01:49:14.187104+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# RX 6900 XT tail: small-batch prefill kernel (tail straggles on short-prompt batches)

## Description

On flashnext-4gpu, six ~900-token prompts submitted together take 2.2 s to the last first token against 1.4 s on the same build without the RX 6900 XT tail (2026-11-01): at ~5.4K-token steps the tail (gfx1030, Gen3 x4 via chipset) is the last helper. Its packed-f16 rows kernel (#486/#488, moe_lane_gemm.h lane_moe_rows_w4) was tuned on 2,048-row steps. A single short prompt is not affected (1K prefill 3.6k tok/s, flat curve to ~150K). Placement is the primary fix (RHP04 stage 3 critical-path solver can give the tail fewer/colder experts); this item covers a kernel variant for small row counts so the tail costs less at small batches whatever it owns.

## Steps

1. Capture tail requests at 6 x short-prompt prefill and replay them (replay-kernel-time.sh) to split act/gemm/write and kernel time by row count.
2. Try a small-row variant of the rows kernel (tile/row-block sizing for <=512 rows per expert, fewer staging passes) on replay.
3. Route by row count; end-to-end A/B on time to first token for 6 x short prompts and on the 42K probe (must not regress).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Replay kernel time per row bucket against the current kernel on identical captures; RHP12 gate including the 6 x short-prompt TTFT and 42K probe; KLD at the chunk minimum.

## Effort & Risk



## Standards



## Acceptance Criteria

Measurable reduction of the 6 x short-prompt time to first token with no regression on long prefill or decode.

## Notes

Kernel changes stay in the bigcherry moe-remote component (managed). Coordinate with RHP04 stage 3: placement may reduce the need.

## Change Log

- 2026-11-02T01:49:14.187104+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_014941_documented-the-r9700-idle-powe_8134
- 2026-11-02T01:49:49.397575+00:00 (updated-by): Updated: section:ledger-events
