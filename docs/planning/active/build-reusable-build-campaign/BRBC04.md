---
id: BRBC04
order: 0
plan: build-reusable-build-campaign
state: pending
created-at: '2026-11-02T08:21:21.252801+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: M
priority: P1
---

# Build system writes its own persistent, consistent per-invocation and per-lane logs

## Description

Owner requirement (2026-11-02): the build system must log consistently. Today `bigcherry build` persists no log of its own.
- Stage output (materialize, generate, configure, compile, runtime-smoke) and the worker exception the failure message tells you to inspect go only to the caller's stdout.
- A failed build leaves just a lock and build-advisories/<run-id or 'unspecified'>/advisories.json. Every untagged run overwrites the same 'unspecified' file.
- Successful build dirs keep only .ninja_log and metadata.
- Observed 2026-11-02: an RSC06 build failed at materialize (rad0010 imported bigcherry.patch.apply). The only record was one line in a run wrapper's stdout, and that wrapper's own log path was broken.

## Steps

1. Every `bigcherry build` invocation gets a run id. Use --run-id or generate one (UTC timestamp plus a short random suffix); never 'unspecified'.
2. Write an invocation log at <work-root>/build-logs/<run-id>/build.log with everything the command prints, plus one file per lane and stage: <run-id>/<lane>/<stage>.log holding the full stdout/stderr of each subprocess (cmake, ninja, patch apply, smoke) and the full traceback of any worker exception.
3. Write a structured summary, <run-id>/summary.json: lanes, stages, status, durations, build_plan_id, build dir, log paths, and the first error line per failed stage.
4. Successful builds also link or copy their stage logs into the build dir as build-logs/, so a build dir explains itself.
5. Advisories are written under the same run id.
6. On failure the final message names the log file holding the exception and the failing stage's log, not just 'inspect the worker exception'.
7. Retention: covered by the existing work-root cleaner (build-run-cleanup), stated explicitly.
8. The lab wrapper rad_build (tools/lab/radiance/lab-build.sh) and other build callers print the run's log directory instead of relying on their own redirect.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Card-free tests:
- a forced materialize failure (a patch with a disallowed import) produces build-logs/<id>/build.log, the failed stage's log with the full traceback, and summary.json naming the failing stage and first error line;
- a forced configure failure and a success both produce stage logs, and the success also has build-logs/ in the build dir;
- two untagged runs do not overwrite each other;
- the final failure message contains the log path.
Real check: one in-container radiance-lab build, success and a deliberate failure, inspecting the files.

## Effort & Risk



## Standards



## Acceptance Criteria

No build stage's output or exception exists only on stdout. Every invocation has a unique run directory with a per-stage log and summary.json. Failure messages point at the exact log. Advisories are never written under 'unspecified'.

## Notes

Pairs with the run-script defect found the same day: a wrapper called `set --` before using $1, so all build logs went to a hidden .build.log. The build system should not depend on callers redirecting correctly.

## Change Log

- 2026-11-02T08:21:21.252801+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_090807_every-build-now-keeps-its-own_5704
- 2026-11-02T09:08:13.207624+00:00 (updated-by): Updated: section:ledger-events
