---
id: PVPS10
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-26T03:11:15.402522+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Trigger-coverage profiling stage: prove the patched kernels run, how much, and bound the expected effect

## Description

Validation today proves only that a patch path ran once (once_flag BIGCHERRY_PATCH_HIT marker). It does not show how much of the workload goes through the patched code, which kernels actually ran, or their share of GPU time, so a neutral verdict cannot distinguish 'kernel not faster' from 'path is 0.3% of GPU time on this workload' (1256/1257), and an implementation that routes to the wrong kernel can pass on marker alone. The profiling tooling exists (bigcherry kernel-fraction over rocprofv3 traces, profile-campaign, 0200 dispatch hook + 0700 coverage counters in the tuning build that every session already compiles but never runs, 0810 replay hit diagnostics) but validation does not use it.

## Steps

1. Patch metadata: each validation package declares its patched kernel families / symbol patterns (validation.toml [profile] kernels=...) and the workload that must exercise them.
2. Stage (per session, untimed, after the verdict lanes): rocprofv3 kernel trace of control and subject on the positive-lane workload; bigcherry kernel-fraction -> per-kernel calls, GPU time and share; subject must show the declared kernels, control must not (or show the pre-patch kernels).
3. Coverage counters: run the scaffold tune build (0200/0700) once for exact per-path hit counts; record hits per token.
4. Expected-effect bound = patched-kernel share x measured kernel-level change (A/B of the kernel alone); record alongside the end-to-end effect.
5. Verdict annotation: patched share below a declared floor => session INVALID (workload does not exercise the patch), not FAIL; large kernel gain with small share => 'not measurable end-to-end on this workload' and a contract review to pick a workload where it matters.
6. Profile artifacts feed improvement work: top kernels before/after, occupancy/launch counts, regressions in neighbouring kernels.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline: kernel-fraction parsing, declared-kernel matching, invalid-vs-fail classification. Hardware: audit jobs on 1256/1257 (top-k), 1263, 1262, 1265-gfx1030 produce traces and shares; 1237/1253 (known winners) show large patched-kernel shares consistent with their gains.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner direction 2026-09-26: 'we should be profiling properly to see if we can improve and validate the patch has been implemented correctly'. First step is a one-off audit via queue jobs (no perturbation of running measurements); the standard stage follows. Related: PVPS04 (tune/replay builds currently unused), PVPS09 (noise), RCD07 (stage split).

## Change Log

- 2026-09-26T03:11:15.402522+00:00 (created-by): Created by agent
