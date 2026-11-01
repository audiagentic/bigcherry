---
id: RHP12
order: 5
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-11-01T22:51:55.015234+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P0
work: M
---

# Every stage: per-layout baselines and the no-regression gate (return to a known layout without degradation)

## Description

Owner invariant 2026-11-01: which cards are helpers and where the engine runs are inputs; changing layout re-adjusts automatically, and switching back to a known layout must reproduce its recorded performance. Tuning or features added for a new layout must not degrade existing ones. This item defines the baselines, the gate and the rollback, and every RHP stage must pass it.

## Steps

1. Record the first baseline: flashnext-4gpu on the W4H2 default (#505) - 8-session prefill probe, 6-stream decode (step time from /metrics), 42K probe, KLD.
2. Record flashnext-2xtx (and other declared layouts available) the same way.
3. A bigcherry command that runs the gate set for a layout and compares against its baseline with paired ABBA statistics.
4. Wire it into each RHP stage's validation.

## Detailed Solution & Technical Design

Baselines under the work root (never in git): $BC_WORK_ROOT/baselines/<layout-fingerprint>/<model-build-fingerprint>/{manifest, prefill-8session, decode-6stream, probe-42k, quality, kernel-times}.json; tuned state under $BC_WORK_ROOT/tuning/{topology,calibration,placement}/<key>.json. Reviewed summaries go to docs/evidence/. The baseline key includes the resolved layout, hardware identity, measured link topology, model and weight digest, backend/kernel build digests, driver/ROCm and serving configuration; the topology key and the benchmark cohort key stay separate. Validated baseline and candidate tune are separate records: a candidate is promoted only after correctness, memory, stress and performance gates pass; a validated baseline is never overwritten by a new tune. On regression the candidate policy is disabled and the last validated ownership restored through a new live epoch (no restart). Return to a known layout: rediscover, reconcile devices, reuse matching calibration and placement records after compatibility checks, stage weights through the live path, commit a new epoch, then check the baseline cohort before promoting.

## Code Samples & Guidance



## Files



## Validation

The gate itself is validated by re-running a baseline layout unchanged (must pass) and with a deliberately worse setting (e.g. the old W4 GEMV) (must fail).

## Effort & Risk



## Standards



## Acceptance Criteria

Baselines exist for every declared layout available on the host; the gate command compares a run against them; every RHP stage cites a passing gate run before merging a default change.

## Notes

Gate (initial proposals, judged against paired variability, ABBA with >=4 clean pairs): median model-step time <=1-2% worse, p95 <=3%, prefill chunk completion <=2%, unchanged kernels <=2%, migration-period p95 <=5% sustained (spikes reported separately), cache-hit start-up <0.5 s, VRAM never above the hard budget, quality within the KLD/top-1 band, zero missing/duplicate/stale-epoch expert work. Step, layer-wait and kernel time are primary; tok/s is not (the engine is not batch-invariant at temperature 0). The 8-session prefill, 6-stream decode and 42K probes are required for any change to placement or its execution path. Plan source: GPT staged plan 2026-11-01.

## Change Log

- 2026-11-01T22:51:55.015234+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261101_225732_planning-a-staged-roadmap-for_4092
- 2026-11-01T22:57:54.713409+00:00 (updated-by): Updated: section:ledger-events
