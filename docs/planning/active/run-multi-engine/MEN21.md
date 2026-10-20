---
id: MEN21
order: 21
plan: run-multi-engine
state: pending
created-at: '2026-10-18T13:04:12.460753+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# moe-harness bigcherry commands: capture, replay sweep, trace report, balance score

## Description

The helper replay harness (r3_moe_worker --replay-sweep, capture format 2) and its drivers are the fast loop for helper work but run as lab scripts. Wrap them as bigcherry verbs on the radiance engine: `moe-capture` (record real requests during an ab cell), `moe-replay` (one model load, env variants, bitwise compare, optional rocprofv3 kernel trace), `moe-trace-report` (TRACE lines to critical-path report), `moe-balance-score` (offline layout scorer). Each writes a JSON record under the run workspace and reads its builds through the lane, not a hand-built tree. Replay is also the identity gate for RCS02-RCS05.

## Steps

1. Move the Python (small_capture, trace_report, balance_score) into tools/bigcherry/... as library code with unit tests.
2. Command drivers call `build --lane`, find the helper binary from the runtime home, check the host is idle, write records.
3. Document in docs/reference/engines/radiance.
4. Retire helper-capture.sh, helper-sweep.sh and the Python scripts; update TOOL_DISPOSITION.
5. Check whether profile-campaign covers fn-profile*.sh for radiance; if not, add a radiance profile mode here.

## Detailed Solution & Technical Design

## Code Samples & Guidance

## Files

## Validation

A replay sweep through the command is bitwise-identical to the script result on the stored captures; offline tests for the report and scorer.

## Effort & Risk

## Standards

## Acceptance Criteria

## Notes

Depends on RCS04 for the final component home but can start against libr3 now.

## Change Log

- 2026-10-18T13:04:12.460753+00:00 (created-by): Created by agent
