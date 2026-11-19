---
id: MRC07
order: 0
plan: run-measurement-records
state: pending
created-at: '2026-11-19T21:40:26.884199+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P3
work: S
---

# llama.cpp producers: remove the run_bench redirect shim and migrate the bench write path (deferred, not Radiance)

## Description

MRC06 slice 2 (#1205) added tools/bigcherry/bench/llamacpp_runner.py, which monkeypatches the external llama.cpp bench/run_bench.py (append_result and its SQLite sync) to redirect results into run-record v1. That is a shim, against the project rule (no legacy, no shims: migrate and refactor up). Deferred 2026-11-20 by owner direction (Radiance only for now).

## Steps

1. Find where run_bench.py lives and who owns it (BC_LLAMACPP_ROOT checkout vs repository code).
2. If we own it: move the measurement write into run_bench itself through core.measurement_producers (one write path), delete its JSON/SQLite result writers and the runner shim.
3. If we do not own it: retire the run_bench measurement path from bigcherry (delete llamacpp_runner.py and its docs/tests) and measure llama.cpp through a bigcherry-owned workload instead.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

No monkeypatching of external code in tools/; llama.cpp measurements (if any remain) write run-record v1 through core.measurement_producers; tests and guards pass.

## Effort & Risk



## Standards



## Acceptance Criteria

llamacpp_runner.py's redirect is gone; exactly one llama.cpp measurement write path or none.

## Notes



## Change Log

- 2026-11-19T21:40:26.884199+00:00 (created-by): Created by agent
- 2026-11-19T21:40:47.843002+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
