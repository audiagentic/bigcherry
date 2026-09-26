---
id: PVPS09
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-26T00:23:54.465222+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Robust paired-lane statistics: single transient rounds must not decide a session

## Description

2026-09-26 noise triage (tools/lab/plan-qualification/noise.py): the only wide-interval sessions of serial-2 each contain ONE catastrophic round -- 1262 gfx1100 s3 pair 4 subject 69.3 vs 99.7 tok/s (-30%), 1263 gfx1100 s2 pair 2 subject -16% -- while their median pair ratios (+0.97%, +0.85%) match the patch's other sessions. The geometric-mean bootstrap over 10 pairs lets one transient (clock/power-state dip) swing a session verdict for any patch.

## Steps

1. Record llama-bench's own per-run stddev per sample; pre-declared re-measure rule: a sample whose internal CV exceeds a declared bound is re-measured once (both arms of that pair, same order), and every re-measure is logged.
2. New effect policy improvement_no_regression_v2: per-session effect = Hodges-Lehmann (median of pairwise ratio averages) with bootstrap CI; applies only to contracts that declare it (new series), never retroactively.
3. Find the transient's cause: log GPU sclk/power/temperature (rocm-smi) around each sample.
4. GPT review of the v2 policy before adoption.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline: synthetic series with one -30% round keeps the session CI near the true effect; hardware: 1263 series 2 on the new policy.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner asked 2026-09-26 whether to retest close calls; answer: no optional stopping, but pre-declared new series are legitimate. 1263 series 2 (20 rounds, contract amended) waits for series 1 gfx1201 s4 to finish.

## Change Log

- 2026-09-26T00:23:54.465222+00:00 (created-by): Created by agent
