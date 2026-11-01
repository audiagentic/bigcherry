---
id: RAD43
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-01T11:26:24.954166+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# Kernel work: helper reduce/write stages, the tail's remaining latency, XTX gate/up/down, engine add

## Description

After #486 the per-request stage times on a 2048-row prefill step are, XTX head: routes ~0.42 ms, gate/up ~1.5, down ~?, reduce/write ~?, total 3.4-3.6 ms for ~9,200 routes; 6900 XT tail: routes 0.54, gate/up 1.53, down 1.25, reduce 0.84 for ~900 routes (1-row experts still latency-bound at ~165 GB/s of weight reads). Engine (R9700, --profile-ops, combine off): moe_mailbox_add 55.6% (wait + host read), dense gemm_nt_q 10.5%, local moe_gemm_q 7.9%, publish 5.3%. Each stage gets a model-free harness (replay-kernel-time.sh on captures, gfx1030/rowsk.hip) before any change.

## Steps

1. Stage table for the XTX helper from replay captures (rocprof): which of route compaction, gate/up, quant, down, reduce dominates and how far each is from its bandwidth or compute bound.
2. The reduce/write stage on every helper: it writes rows over the link; check it achieves the link's write rate and overlaps with the down GEMM tail.
3. The tail's 1-2-row experts: GEMV path or merged small experts in one block, measured in rowsk.
4. Engine add kernel (moe_mailbox_add_rows): achieved host-read GB/s against host-link-bw's pull rate.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per-stage harness numbers before and after each change; live 42K probe and 8-session run for every kept change; bit-exact or KLD inside the noise band.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Measurement first; only changes with a harness-measured gain go on.

## Change Log

- 2026-11-01T11:26:24.954166+00:00 (created-by): Created by agent

## XTX head, replay against live (2026-11-01, `runs/rad43-capx`)

60 captures of helper 0's large-lane requests in the 42K probe (flashnext-4gpu), 20 replayed under rocprof on the XTX: the
grouped GEMMs (gate/up and down, the unnamed kernel) 2 x 1,203 us = 2.41 ms a request, the reduce/write 722 us, the
activation copy 383 us; live stage times 1,495 + 867 = 2.36 ms, 669, 429. So the head is kernel-bound, no contention gap
(unlike the tail's). A request: ~9,300 routes over ~157 experts (rows an expert: 1-row 11, 2 8, 3-4 11, 5-8 15, 9-16 19,
17-32 24, 33-64 28, 65+ 41) = ~46 GFMA in 2.4 ms, ~19 TFMA/s, about a third of the XTX's matrix-core rate; weights
(~0.4 GB) would take ~0.4 ms. Levers: experts of <=16 rows (64 of 157) fill 16-row WMMA tiles mostly with padding, ~10%
of the tile work (send them to the few-rows kernel within the same request: ~0.25 ms); the grouped GEMM's own efficiency
(the larger prize, radiance's gfx1100 MoE GEMM). Heads are the last helper on 62% of prefill layers.

Tile knobs on the same 20 head captures (rocprof total of the grouped GEMMs, all bit-identical): default 93.7 ms; MOE_FORM=1
93.6; MOE_FORM=2 103.6 (+11%); MOE_LONG_ROWS=32/64/128 108.0/108.0/103.1 (+10-15%); MOE_DOWN_TILE=2 fails on this model
(libr4d moe_gemm_w4a8 launch rc=-18, the replay then waits forever). The default is already the best of the plugin's knobs:
a head gain needs new kernel work (few-row experts to the few-rows kernel in the same request; the grouped GEMM itself).

## Where the XTX grouped GEMM's time goes (2026-11-01, scoping, `runs/rad43-capx/pr-*`)

The kernel is libr4d's `moe_gemm_q_kernel` (radiance 1.14.0 `libr4d/r4d_moe.hip`, ~1,100 of its 3,800 lines; templated on
MT/NTL/WGM/WGN, the w4nl path decodes weights to f16 and multiplies with f16 WMMA). Its timing probe (R3_MOE_PROBE, outputs
invalid) on the same 20 head captures: normal 93.6 ms; without the WMMA instructions 76.7 ms (-18%); without widening the
E4M3 activations 77.8 ms (-17%). So about two thirds of the kernel is neither the matrix math nor the activation widening:
weight decode, LDS staging, global loads and barriers. A rewrite would target the weight/staging path (e.g. decoding w4nl
straight into WMMA fragments, double-buffered weight tiles, fewer barriers), not the arithmetic. Size: L (one large kernel
family, five template axes, two weight formats); expected gain if the non-math part halves: ~1/3 of the head GEMM, ~0.8
ms a head request, the heads being last on 62% of prefill layers.

