---
id: RSC02
order: 2
plan: run-serving-configuration
state: pending
created-at: '2026-11-02T00:53:02.723581+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P0
work: M
---

# bc-serve: the only start-up path (detect, load, write manifest, start processes with --manifest only)

## Description

Replace tools/lab/radiance/launch-flashnext-best.sh with `bigcherry serve MODEL [--config serving.toml] [--profile ...] [--set key=value] [--explain --dry-run]`, driven by the RSC01 resolver. Layouts become optional constraints resolved against discovered hardware, not required hand-written topologies. Engine arguments and per-helper environments are generated from the effective manifest through an allowlist, so inherited stray R3_* variables cannot change a run. Unique shared-memory identity per instance, supervised helper processes, readiness checks, stop-everything-on-partial-failure, signal handling, conflict detection. The lab A/B runner resolves each arm through the same code. Raw --engine-arg pass-through is removed.

## Steps

1. `bc-serve [--profile p] [--layout l] [--set k=v] [--dry-run]`: detect (RSC05), then load and validate (RSC01), then write the immutable manifest to the run directory, then start the helpers and the engine with only `--manifest <path> --role <role>`, then health-check, then record (manifest hash and process PIDs).
2. Supervision: ordered start (helpers ready before the engine), graceful stop, PID files, and failure propagation (one process failing stops the set).
3. Delete tools/lab/radiance/launch-flashnext-best.sh and every caller of it. Replace the RSC01 launcher-equivalence test with a manifest golden test.
4. Lab runs use `serve` with experiment-namespace overrides, not separate scripts.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

The first version reproduces the current flashnext-4gpu baseline exactly (same manifest, RHP12 gate); lifecycle tests: engine or helper failure, partial start, repeated start/stop, IPC isolation, inherited-environment isolation; experimental settings refused by serve, allowed by the lab runner with evidence isolation.

## Effort & Risk



## Standards



## Acceptance Criteria

One command starts Flash-Next on any supported layout. No process receives configuration through flags (other than --manifest/--role) or R3_* environment variables. The bash launcher is gone. A dry run prints the manifest and the process plan without starting anything.

## Notes

Depends on RSC01. Source: GPT configuration assessment 2026-11-02.

Depends on RSC01 (loader), RSC06 (helpers read the manifest) and RSC07 (engine reads the manifest). Owner decision 2026-11-02 (recorded on RSC01): no command-line converter and no fallback path.

## Change Log

- 2026-11-02T00:53:02.723581+00:00 (created-by): Created by agent

## Ledger-events


- chg_20261102_005338_planning-one-serve-command_5259
- 2026-11-02T00:53:48.446911+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-02T03:51:40.629364+00:00 (updated-by): Updated: section:title, section:steps, section:acceptance_criteria, section:notes
- chg_20261102_035356_serving-start-up-plan-one-str_4985
- 2026-11-02T03:54:04.045012+00:00 (updated-by): Updated: section:ledger-events
