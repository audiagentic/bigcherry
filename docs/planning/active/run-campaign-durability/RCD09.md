---
id: RCD09
order: 9
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:20.106874+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Job service observability: watch stream, wake events, status schema, report view, GPU telemetry

## Description

bigcherry jobs watch --jsonl --after <seq> [--wake-only]; stable event kinds and severities; default wake events (attempt.failed-harness, attempt.stalled, job.contract-drift, job.composition-drift, host.disk-hard, host.gpu-unhealthy, harvest.failed, daemon.recovery-ambiguous, series.complete) vs recorded-only (started, phase changes, done-pass, done-fail, ladder); bigcherry jobs status --json schema (queue depth, running attempt/phase, per-GPU claims, failure counts by kind, disk/ccache, last events); minimal report/dashboard view; GPU telemetry (sclk/mclk/temp/power/util/VRAM) per sample, record-only until PVPS09.

## Steps

1. Event schema + watch.
2. status --json.
3. Telemetry collector.
4. Replace hand-written watcher loops for agents.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline watch-resume and schema tests; an agent session wakes on a forced harness failure without a shell loop.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-26T00:52:20.106874+00:00 (created-by): Created by agent
