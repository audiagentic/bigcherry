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



BACKLOG TRIAGE 2026-11-04 (MERGE into RAD43; revalidated against main 6eb31a708, verdict CONFIRM): RAD43 (helper kernel hot-path profiling) and RCS08 exist and are open; capture/replay tooling already in tools/lab/radiance and worker --replay. Evidence: RAD43 (helper kernel hot-path profiling) and RCS08 exist and are open; capture/replay tooling already in tools/lab/radiance and worker --replay. Still referenced by: RAD10, MEN14, WSL05. Original content kept below for history.



REOPENED 2026-11-04: an independent revalidation (against main 6eb31a708) rejected the closure recorded above, which had been based on a lighter check. It stays open. Why: MEN21.md:18-31 requires moe-capture/replay/trace-report/balance-score verbs, build-lane binding, bitwise sweep and script retirement; no corresponding command definitions appear under tools/bigcherry. RAD43.md:18-30 instead owns replay-based GPU kernel-stage optimization, and RCS08.md:18-27 owns component lint, not the generic operator harness.

## Change Log

- 2026-10-18T13:04:12.460753+00:00 (created-by): Created by agent
- 2026-11-04T03:47:35.947649+00:00 (updated-by): Updated: section:notes
- 2026-11-04T03:47:39.960586+00:00 (state-transition): State: pending → superseded
- 2026-11-04T04:45:30.747955+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:45:34.908530+00:00 (state-transition): State: superseded → pending
