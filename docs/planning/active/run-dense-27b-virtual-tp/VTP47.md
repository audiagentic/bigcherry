---
id: VTP47
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-23T20:42:19.617288+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# Small-shape iq8r decode GEMV on gfx1100: split-K / activation-in-LDS skinny form for the rows far below stream rate (attention k/v, attention out, delta-net out, MLP down)

## Description

Source: VTP37 decode GEMV evidence (2026-11-23): at rank 0's real shapes `gemm_iq8r_nt_m16` runs far below the stream rate on several rows: attention k/v 256x5120 at 9% of bandwidth (16.0 us), delta-net out 5120x2304 45%, attention out 5120x1536 38%, MLP down 5120x6656 64%, delta-net in 66%, against 78% for MLP gate/up. Lifting the sub-target rows to 650 / 751 GB/s saves 0.88 / 1.84 ms a step (4.0 / 8.3%). VTP37 hypothesised these small shapes are launch- or tail-bound and need a different kernel form (split-K, fusion). A reference decode GEMV set shows the usual skinny forms: activation vector staged once in 64 KB of LDS, 16-byte weight chunks streamed with an unroll of 2, the K range split across waves with a DPP / LDS reduction, wave counts tied to N so a small N still fills the device. Our wave-count and token-tile pins do not reach the K-parallel decode tile, so this is not yet tested.

## Steps

1. Per-row occupancy model at the six rank-0 shapes: workgroups launched against the XTX's 96 CUs and the waves per SIMD each tile gets (the k/v row at N 256 launches very few).
2. Scratch split-K variant of the K-parallel tile in libr3 native: K split across extra workgroups or waves, partials reduced in registers/LDS in a fixed order (integer group sums are exact, so int8 partials add without changing the result); activation tile in LDS where it removes redundant loads.
3. Cold timing replay at the six shapes (M 1-16 and 17-32) with the scratch driver; byte identity against the shipped row.
4. Decode ms/step ABBA at 1, 6, 8 streams; acceptance and temp-0 text unchanged.

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



## Change Log

- 2026-11-23T20:42:19.617288+00:00 (created-by): Created by agent
- 2026-11-23T20:43:12.216774+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
