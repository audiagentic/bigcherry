---
id: ENG01
order: 0
plan: tuning-radiance-engine
state: pending
created-at: '2026-11-10T13:11:33.523674+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# R9700 engine per-op attribution with the live policy on: prefill (document, cold) and decode, with an unprofiled reference

## Description

First kernel-cost decision gate for production Radiance on the R9700 engine with two XTX helpers. Attribute exposed per-operation work on the current source/ABI with live policy confirmed ON, and pair the profile with a matched unprofiled baseline. Existing historical local-call and decode figures are diagnostic context, not proof of current policy-on op shares or prospective speedup. Profile instrumentation can synchronize and disable replay; keep its overhead separate.

## Steps

1. Record exact current source/ABI/build, topology, manifest and runtime feature state; verify live placement is ON and identity matches the production R9700-engine/two-XTX-helper layout.
2. Profile separate cold/document prefill and one-/six-stream decode probes. Correlate rank-0 operator time, host work, synchronization, overlap and idle wait using valid clock domains; identify exposed critical-path time rather than summing overlapped device time.
3. Run the same production configuration unprofiled for throughput and dispersion. Keep instrumentation overhead and any admission/resource changes explicit.
4. Rank operations by exposed time and calculate a maximum Amdahl bound for eliminating each operation. Compare it with paired noise and a declared worthwhile gain before admitting a kernel candidate.
5. Record the go/no-go and route any admitted single-operation work to its existing owner. Do not create a generic optimization campaign.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Policy-on identity and runtime activation are recorded with source/build/ABI, model, topology and measured-plan key. Per-op tables cover cold/document prefill and one-/six-stream decode; matched unprofiled references report dispersion. Correlated timestamps distinguish exposed CPU wait from overlapping GPU event totals; diagnostic overhead is stated.

## Effort & Risk



## Standards



## Acceptance Criteria

A current production R9700-engine/two-XTX-helper per-op attribution and matched unprofiled baseline are recorded. The exposed path, operator shares and maximum Amdahl bound are explicit. Each kernel candidate has a written admission/rejection against paired noise and a declared minimum worthwhile gain. No kernel implementation is admitted without this gate.

## Notes

Source: gateway prior-work audit 2026-11-10; my first attempts: runs/engine-profile-policy-on.

## Change Log

- 2026-11-10T13:11:33.523674+00:00 (created-by): Created by agent

## Reviews

- RV4235
- 2026-11-20T02:19:41.977225+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
