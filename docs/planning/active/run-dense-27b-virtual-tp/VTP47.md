---
id: VTP47
order: 0
plan: run-dense-27b-virtual-tp
state: in_progress
created-at: '2026-11-23T20:42:19.617288+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# Small-shape iq8r decode GEMV on gfx1100: split-K / activation-in-LDS skinny form for the rows far below stream rate (attention k/v, attention out, delta-net out, MLP down)

## Description

VTP37's decode GEMV evidence (2026-11-23): at rank 0's real shapes `gemm_iq8r_nt_m16` runs far below the stream rate on several rows: attention k/v 256x5120 at 9% of bandwidth (16.0 us), delta-net out 5120x2304 45%, attention out 5120x1536 38%, MLP down 5120x6656 64%, delta-net in 66%, against 78% for MLP gate/up. Lifting the sub-target rows to 650 / 751 GB/s saves 0.88 / 1.84 ms a step (4.0 / 8.3%).

**Occupancy model (read from our launcher, 2026-11-24).** The K-parallel int8 tile launches one workgroup of at most four waves per 16 weight rows, with the waves striding K by scale group. Workgroups therefore equal N / 16 and waves equal 4 N / 16: attention k/v (N 256) launches 16 workgroups, 64 waves, on a 96-CU card; attention q/gate (N 3072) 192 workgroups; N 5120 rows 320; MLP gate/up (N 13312) 832. Stream rate rises with that count (9% at 16 workgroups, 38-66% at 192-384, 78% at 832), so the small rows are limited by how few waves are in flight and by the serial load chain each wave walks over K, not by the tile arithmetic.

**What we will try, in order.** (1) More waves per workgroup where K allows (eight instead of four): a pure launcher-side test, measured first because it needs no kernel change. (2) Splitting K across workgroups for the rows that stay under about 300 workgroups, with partial sums combined in a fixed order that reproduces the current reduction order so results stay byte-identical (the int32 group sums are exact; only the f32 combine order matters). (3) Fusing the tiny attention k/v row with the neighbouring q/gate row into one launch. Each step is kept only if the measured isolated time and then the serving step improve.

## Steps

1. Scratch comparison, cold weights, one XTX, at the six rank-0 shapes and M 4, 16, 32: current launcher against eight waves per workgroup (scratch builds under the task tmp folder, timing with the scratch driver, three repeats). Started 2026-11-24.
2. If wave count helps, find the rule (keyed on N and K, never a model) and check byte identity of the outputs against the shipped row at M 1-64.
3. K split across workgroups with an order-preserving combine, for rows still under 300 workgroups; fusion of k/v with q/gate if the launch gap remains.
4. Decode ms/step ABBA at 1, 6 and 8 streams; acceptance and temp-0 text unchanged.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Byte identity of the int32 sums and final outputs at M 1-64 at all six shapes.
- Resource record: no spills, no new instantiation above 256 VGPRs.
- Serving ABBA (untraced, 3 seeds) at 1, 6, 8 streams.

## Effort & Risk

Medium. Risk: extra reduction work and launches cancel the gain for the tiniest shapes (k/v), which may need fusion with the neighbouring q/gate rows instead.

## Standards



## Acceptance Criteria

- Each shape moves to at least 650 GB/s where the shape allows it, or a recorded NO-GO per shape; the step-level gain is measured, not inferred from the isolated bound.

## Notes

2026-11-24: occupancy model written above; step 1 builds made (baseline and eight-wave variants of the K-parallel tile) and timing started under the gpu lease.



**2026-11-24 step 1 result (eight waves per workgroup against four; one XTX, cold weights, scratch driver, three repeats, median of 30; the driver's time includes about 10 us of launch).** Six rank-0 shapes, M 4 / 16 / 32, change in time: MLP gate/up -0.8 / -0.9 / +1.8%; MLP down -5.7 / -7.2 / -1.5% (611 -> 648 GB/s at M 4); delta-net in -1.6 / -3.0 / +2.8%; attention q/gate +1.3 / +4.6 / +2.8%; delta-net out -2.8 / -1.0 / 0.0%; attention out +1.9 / 0.0 / +1.5%; attention k/v -12.8 / -14.9 / -16.6% (14.8 -> 12.9 us at M 4). Reading: more waves per workgroup is not the lever for the large rows, and the rows that look worst by stream rate (k/v 89 GB/s, attention out 367, delta-net out 415) sit near the launch-and-latency floor of the driver: k/v is 12.9 us of which about 10 us is launch, so its isolated bound overstates what a step can gain. The step-level ceiling of this knob is about 0.1 ms, so it is not worth shipping alone. The earlier 0.88 / 1.84 ms isolated bounds should be re-read with the launch term removed before further GEMV work: next is to measure the same rows inside a replayed graph (launch gaps there are far smaller than the driver's 10 us) or take per-op times from the step trace. Steps 2 and 3 (K split across workgroups, k/v + q/gate fusion) stay open but pay mainly on launch count, so they move behind a launch-count audit of the pass.

## Change Log

- 2026-11-23T20:42:19.617288+00:00 (created-by): Created by agent
- 2026-11-23T20:43:12.216774+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-23T20:48:37.567386+00:00 (state-transition): State: pending → in_progress
- 2026-11-23T20:54:39.540860+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-11-23T20:55:50.196625+00:00 (updated-by): Updated: section:notes
