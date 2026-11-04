---
id: RAD48
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-04T18:09:26.044198+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Native gfx11 int8 prefill tile (27B INT8 on 7900 XTX)

## Description

gemm_i8a8_tiled ran at ~28 TFLOP/s through the compatibility layer (16 cross-lane moves per iu8 WMMA). Native form: accumulator stays in the gfx11 layout through a 128-K block and is converted once per block; activation fragment read from LDS as a gfx11 row; weight fragment converted once per K step. Measured on one 7900 XTX (rad-kbench, fixture q27-w8a8-tok2048, N=34816 K=5120): M=2048 25.8 ms -> 8.33 ms (~88 TFLOP/s, 3.1x); M=128 ~1.6 ms -> 0.56 ms; all i8a8h cases within tolerance of libref (worst rel_l2 2.2e-5). The 600 failures in the all-rows run are libr3 gemm_iqr_* rows refusing i8a8h (dtype this kernel does not serve), unrelated and pre-existing. Open: M 33-64 crossover between gemm_i8a8_nt_m16 (358 us at M=32, 772 us at M=64) and the tiled row (~550 us flat); end-to-end 27B INT8 prefill and KLD check; IQR row constraint mismatch.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

engines/radiance/components/libr3/native/r4d_gemm_8a8_prefill.h.rw

## Validation

rad-kbench --all-rows replay of the INT8 fixture against libref; end-to-end 27B INT8 prefill tok/s on 2 XTX before/after under the gpu lease; KLD vs previous build.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-11-04T18:09:26.044198+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261104_180932_qwen38-27b-int8-prefill-gemms_1345
- 2026-11-04T18:09:36.325907+00:00 (updated-by): Updated: section:ledger-events
