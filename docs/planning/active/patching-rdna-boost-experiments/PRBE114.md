---
id: PRBE114
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-29T23:14:03.823307+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# 1263 prbe41 tensor-split: instrument SSM_CONV split state and propagate d_inner split through the channels-major reshape

## Description

GPT review 2026-09-30: 1263 dies in meta allocation before its marker (GGML_ASSERT(src_ss[0].nr[0] == 1)), so it is not evidence that channels-major CUDA is bad. The package already maps src0 axis-0 + src1 axis-1 -> output axis-0, so a source reaching SSM_CONV still has nr[0] != 1: a deeper split-state/reshape propagation problem. Instrument handle_ssm_conv with each source's {axis, nr, ratios} and tensor shapes, then teach meta reshape/view propagation to preserve the d_inner split rather than special-casing the terminal assertion. Until fixed, 1263 is single-device only, not tensor-split compatible.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

27B Q8_0 under -sm tensor starts, marker fires, output matches stock; then 4-session contract campaign.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Queue batch t-1263-gfx1100-s* exercises the single-GPU contract meanwhile.

## Change Log

- 2026-09-29T23:14:03.823307+00:00 (created-by): Created by agent
