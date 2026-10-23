---
id: RAD05
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-14T18:52:42.248696+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
---

# gfx11 decode MoE GEMV for w4nl64a8h (helper and XTX engine expert GEMMs at decode size)

## Description

Measured 2026-10-15 (RAD02 helper per-stage timing, XTX helper serving the R9700 engine, decode-size requests <= 8 rows, full-width experts): gate/up MoE GEMM 242-259 us and down 102-108 us of a ~455 us request. libr4d's decode form of the same GEMM measures 36 us for gate/up at 40 rows on gfx1201 (tp2 half width), so the XTX path is ~4x off its weight-bandwidth bound. Cause: gfx1100 has no fp8 WMMA; libr3 runs the w4nl64a8h MoE kernel through bf16-widened WMMA emulation (R3_MOE_NATIVE=1 only -6%). The helper's per-layer round trip is the decode limit of the R9700-engine layout (R9700 profile: moe_mailbox_add 33%).

## Steps



## Detailed Solution & Technical Design

A decode-sized MoE GEMV for gfx1100 (libr3 native unit), registered as a moe_gemm_q row for dtype w4nl64a8h banded to small T, used by the helper's small lane and the XTX engine's MoE: no WMMA; each wave owns a (sorted route row, block of N) tile, streams the expert's packed nibble rows once, decodes codes through kW4nlTable (libr4d/r4d_args.h) with the E4M3 per-(row, 64 of K) scale times kW4G64Scale, against the E4M3 activation with its f32 per-128-of-K scale (Hadamard baked into both), f32 accumulate, bf16 out. Same moe_gemm_q ABI (Aq, As, code/scale tables per parity class, sorted, off, out; a_order token for gate/up, sorted for down). Must honour moew4_permute's nibble order and the scale plane layout (r4d_layout.cpp).

## Code Samples & Guidance



## Files



## Validation

CPU reference vs existing moe_gemm_w4a8 on gfx1100 for 1-8 tokens x top 10 (r3_moe_worker --compare and a libr3 selftest case); KL vs the XTX-pair exact reference within the noise floor (0.058-0.067); helper per-stage timing at decode size; R9700-engine + two-XTX decode A/B (balanced, same prompt, MTP acceptance compared); two-XTX engine decode A/B.

## Effort & Risk

Packed nibble permutation and scale layout must match the relayout exactly; a wrong decode is caught by the CPU compare. Expected: helper decode request ~455 -> ~150-200 us; the XTX engine's decode MoE uses the same kernel.

## Standards



## Acceptance Criteria



## Notes

Follows RAD02 (R9700 engine + two XTX helpers, heat ownership: prefill 2,767 tok/s at 20 GiB helpers, decode ~69.7 at 22 GiB).

2026-10-15 07:20 DONE as helper small-lane kernel (lane_moe_gemv in moe_remote_gpu.h, e4d95b98 + 883ac30d part-pitch fix, default on). Compare: gate/up matches the grouped GEMM to 1e-9 / 4.6e-6; end-to-end compare 6.4e-4 / 1.9e-4 (unchanged). Balanced A/B, R9700 engine + two XTX helpers (22 GiB, heat ownership, small lane 8): off prefill 2515/2531 decode 69.6/69.7; on prefill 2197(warm-up)/2524 decode 95.7/95.5. Helper decode request: gate_up 241->60-67 us, down 102->24-29 us. KL: KLD 0.0605 / top-1 88.62% / PPL 9.003. Remaining scope: use the same GEMV for the XTX engine's own decode MoE (two-XTX layout) and the R9700 engine's small steps; RDNA2 (gfx1030) build for a 6900 helper.

## Change Log

- 2026-10-14T18:52:42.248696+00:00 (created-by): Created by agent
- 2026-10-14T20:14:07.003004+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261014_204134_flash-next-decode-on-dev-re_3947
- 2026-10-14T20:41:38.358166+00:00 (updated-by): Updated: section:ledger-events
