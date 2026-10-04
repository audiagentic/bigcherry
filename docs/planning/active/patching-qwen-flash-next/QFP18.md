---
id: QFP18
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-04T07:54:10.353678+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# Qualify the Flash-Next production stack to validated (after the pin bump)

## Description

None of the 2026-10 Flash-Next patches is 'validated': each needs a qualification package (experiment contract binding, validation.toml wired to the existing ABBA/ab-benchmark tooling, contract evidence on the current pin). Do it after the llama.cpp pin bump (upstream #29825/#29824/#29856) so evidence is bound to the new pin.

## Steps

1. Bump pin (bump-llamacpp skill), re-anchor, re-screen v5 (v4 + 1326 + 1237/1265/1253 if clean).
2. First candidates: 1326_sched_async_host_inputs (+8% t/s, greedy identical) and the Q8_1 launch-reduction stack 1235/1307/1309/1310/1311/1312/1313 (+ 1308 rollback copies); then 1302/1303 (240K f16 enablers) and 1291/1292/1294/1297.
3. Per patch: experiment contract (hypothesis, workload = Flash-Next v5 decode ~10K/~80K, controls, thresholds, activation evidence, greedy identity), validation.toml adapter, README; run contract sessions (balanced, multi-request); patch-verify-evidence; deliberate promotion via the lifecycle skill.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

patch-verify-evidence passes on the new pin; lifecycle promotion recorded with evidence.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner agreed 2026-10-04: qualify after the bump. Current states: evaluated (1302, 1303, 1307-1313, 1326, ...), rejected 1304; 1237/1265/1253 already validated but only now entering the Flash-Next profile.

Correction: 1237/1265/1253 are already in every Flash-Next build via the validated-enhancements patch-set; v5 = v4 + 1326.

## Change Log

- 2026-10-04T07:54:10.353678+00:00 (created-by): Created by agent
- 2026-10-04T07:55:43.580424+00:00 (updated-by): Updated: section:notes
