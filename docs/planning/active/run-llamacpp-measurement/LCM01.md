---
id: LCM01
order: 0
plan: run-llamacpp-measurement
state: pending
created-at: '2026-11-19T22:16:52.347004+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P3
work: M
---

# llama.cpp producers on run-record v1: one owned write path, no run_bench redirect shim

## Description

Carries the llama.cpp half of the run-measurement-records work (MRC06, formerly MRC07), moved out on 2026-11-20 when the owner scoped current work to Radiance only. Work done so far is preserved: merged #1205 added tools/bigcherry/bench/llamacpp_runner.py (monkeypatches the external bench/run_bench.py append_result and SQLite sync to redirect rows into run-record v1, a shim against the no-legacy/no-shims rule) and experiment-bundle/VA25 attestation folding; tag archive/mrc06-llamacpp-producers (d1b97e011) holds the unmerged `bigcherry results run-bench` command and its TOOL_DISPOSITION rows.

## Steps

1. Find where run_bench.py lives and who owns it (BC_LLAMACPP_ROOT checkout vs repository code).
2. If we own it: move the measurement write into run_bench itself through core.measurement_producers (one write path), delete its JSON/SQLite result writers and llamacpp_runner.py.
3. If we do not own it: retire the run_bench measurement path from bigcherry (delete llamacpp_runner.py and its docs/tests) and measure llama.cpp through a bigcherry-owned workload.
4. Reuse what is useful from archive/mrc06-llamacpp-producers, without its compatibility rows.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

No monkeypatching of external code in tools/; llama.cpp measurements write run-record v1 through core.measurement_producers or are retired; tests and guards pass.

## Effort & Risk



## Standards



## Acceptance Criteria

Exactly one llama.cpp measurement write path (or none), no redirect shim, no legacy result writers.

## Notes



## Change Log

- 2026-11-19T22:16:52.347004+00:00 (created-by): Created by agent
- 2026-11-19T22:17:50.065159+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
