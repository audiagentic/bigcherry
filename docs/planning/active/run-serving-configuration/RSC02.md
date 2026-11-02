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

# bigcherry serve: one lifecycle command replacing the shell launcher, generated process environments

## Description

Replace tools/lab/radiance/launch-flashnext-best.sh with `bigcherry serve MODEL [--config serving.toml] [--profile ...] [--set key=value] [--explain --dry-run]`, driven by the RSC01 resolver. Layouts become optional constraints resolved against discovered hardware, not required hand-written topologies. Engine arguments and per-helper environments are generated from the effective manifest through an allowlist, so inherited stray R3_* variables cannot change a run. Unique shared-memory identity per instance, supervised helper processes, readiness checks, stop-everything-on-partial-failure, signal handling, conflict detection. The lab A/B runner resolves each arm through the same code. Raw --engine-arg pass-through is removed.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

The first version reproduces the current flashnext-4gpu baseline exactly (same manifest, RHP12 gate); lifecycle tests: engine or helper failure, partial start, repeated start/stop, IPC isolation, inherited-environment isolation; experimental settings refused by serve, allowed by the lab runner with evidence isolation.

## Effort & Risk



## Standards



## Acceptance Criteria

serve replaces the launcher for every caller (lab scripts migrated, no shim); identical baseline; lifecycle tests pass.

## Notes

Depends on RSC01. Source: GPT configuration assessment 2026-11-02.

## Change Log

- 2026-11-02T00:53:02.723581+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_005338_planning-one-serve-command_5259
- 2026-11-02T00:53:48.446911+00:00 (updated-by): Updated: section:ledger-events
