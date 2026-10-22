---
id: RAD18
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-21T23:17:12.131804+00:00'
breadth: ''
skill: ''
created-by: agent
---

# hc_up_mix_wide_kernel PF2 RT sweep (occupancy/register pressure)

## Description

Per GPT dev-agent review, sweep hc_up_mix_wide_kernel's RT (rows per workgroup, selects MW waves-per-block) at the PF2 call site to find whether occupancy/register pressure (not grid fill, which is already ample at this shape: n/kWideCH=160 column tiles) leaves real headroom in the existing fused BF16 up-mix kernel. Follows the RAD17 strategic pivot (GEMM-only and GEMM+separate-epilogue diagnostics both showed the naive hipBLASLt-replacement approach loses to the fused kernel).

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Context: this is a sibling investigation to RAD17 (FP8 down-product), targeting the BF16 up-mix kernel instead. See RAD17's notes for the full GEMM/epilogue diagnostic history that led here. Patch: rad0008_hc_upmix_rt_sweep, default-off via RAD_HC_UPMIX_RT env var (unset = bit-identical to current production RT=kPf2UpRT=256).

## Change Log

- 2026-10-21T23:17:12.131804+00:00 (created-by): Created by agent
