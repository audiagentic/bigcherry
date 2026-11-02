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

REVISION 2026-11-02 (GPT configuration assessment, reviewed): (1) Scope of 'return to a known layout': within the same helper set and engine card, a previous expert distribution is restored live through a new placement epoch (RHP11 v23). Returning to a layout with a different helper count, device assignment or engine role is a reconfiguration (restart) until a larger protocol exists; the guarantee there is performance reproducibility of the cached effective configuration against its baseline, not a restart-free change. (2) Baselines need two identity dimensions: the topology/model/build key (which calibration is reusable) and the workload cohort (which baseline is comparable). (3) A retained-session cohort is mandatory alongside fresh-server baselines: run the ~1M-token 8-session fill first, then the prefill probe on new text (not a prefix-cache hit) and 6-stream decode; record retained tokens, host KV usage, n-gram placement, engine KV loan size, engine expert residency, host-streamed expert units, mover host-to-device bytes, free host RAM and per-step latency. Found 2026-11-01/02: on flashnext-4gpu with the defaults, retained sessions halve prefill (3,837 -> 1,948 tok/s) and slow 6-stream decode 46.9 -> 53.0 ms because the KV cache's expert loan collapses; fresh-only baselines missed it. (4) Each probe gets a defined server state (fresh server, or explicit order) - probe order changed results by up to 2x. (5) Resource gate: a configuration must keep the OS/cgroup/pin safety reserve; a run that leaves 0 GiB free RAM fails regardless of speed. Effective configuration manifests (RSC01) are recorded with every baseline.

## Change Log

- 2026-11-01T22:51:55.015234+00:00 (created-by): Created by agent

## Ledger-events



- chg_20261101_225732_planning-a-staged-roadmap-for_4092
- 2026-11-01T22:57:54.713409+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-02T00:53:25.263305+00:00 (updated-by): Updated: section:notes
- chg_20261102_005338_planning-one-serve-command_5259
- 2026-11-02T00:54:00.299817+00:00 (updated-by): Updated: section:ledger-events
- chg_20261102_012258_long-running-flash-next-server_6339
- 2026-11-02T01:23:06.428488+00:00 (updated-by): Updated: section:ledger-events
