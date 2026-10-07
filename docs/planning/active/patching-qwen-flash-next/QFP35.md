---
id: QFP35
order: 35
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:37.142989+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Hyper-connection kernels: no 64-bit div/mod, and audit our fused chains against the reported ones

## Description

External report: (a) replacing emulated 64-bit integer division per element with a 3-D grid in hyper-connection kernels took 230 -> 79 us; (b) fusing scale/sigmoid/scale/hc_post and scale/silu each into one kernel removed ~380 kernels per token (38.5 -> 40.6 t/s). We have 1311 (hc_pre_q81) and 1313 (scale_act_fuse) validated; check what is still unfused and whether our hc kernels index with 64-bit div/mod.

## Steps

1. Read the hc kernels for div/mod on 64-bit indices. 2. Per-token kernel census of the hc chain on production: which links are separate launches. 3. Patch each as its own change. 4. Equivalence + ABBA.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Bit-identical for the indexing change; equivalence test for any new fusion (see FKE01); decode and prefill ABBA.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-07T00:39:37.142989+00:00 (created-by): Created by agent
