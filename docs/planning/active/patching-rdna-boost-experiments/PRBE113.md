---
id: PRBE113
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-29T23:14:00.745535+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# 27B dual-XTX profile-first: rocprofv3 decode dispatch capture + 1245 width-6 fused vs unfused + RD13 fusion-hit counter

## Description

GPT review 2026-09-30: three measurements resolve the largest uncertainties without another broad A/B. (1) One steady-state rocprofv3 --kernel-trace --stats capture of stock+narrowed-1241 production MTP on Qwen3.8-27B Q8_0 dual XTX (-sm tensor, n_max=4), aggregating kernel duration/count separately for ncols=1 and verify widths 2-5, plus MMVQ/quantize, GDN, collectives, GLU, copy/CONT, tiny-launch families. (2) 1245 gp11: width-6 fused vs unfused Q8_0 MMVQ kernel duration/count under rocprof. Code-object analysis already ruled out spill (fused ncols=6 uses 60 VGPR, 0 spill, 0 scratch vs 44 VGPR unfused), so the -6.4% at n_max=5 is not register spill; occupancy at 60 VGPR should be unaffected on gfx1100, cause unknown. (3) 1206 rd13: add a graph fusion counter (candidate MUL_MAT->RESHAPE/VIEW->ADD triplets vs fusions per eval, split prefill/decode, GDN vs attention) and run it on the 27B; zero means model-graph mismatch.

## Steps

1. Script under tools/lab/rocprof-27b-decode/ with README, run through the queue rather than ad hoc. 2. Capture on Brutus (GPUs idle, after the current batch). 3. Analyse kernel fractions per width. 4. Decide 1245 reject/fix and rank candidates from the data.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Committed capture summary; per-width kernel time table; 1245 and 1206 decisions recorded with evidence.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Do not run while the gfx1100 queue batch is executing (host-exclusive).

## Change Log

- 2026-09-29T23:14:00.745535+00:00 (created-by): Created by agent
