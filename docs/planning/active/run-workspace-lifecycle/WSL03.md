---
id: WSL03
order: 3
plan: run-workspace-lifecycle
state: pending
created-at: '2026-10-19T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Retention and managed cleanup (`bigcherry clean`, dry run first)

## Description

Automatic cleanup that cannot delete anything we still need. `bigcherry clean` reads the manifests and applies a retention policy per kind: the raw bulk of a HARVESTED run is deleted after N days
(default 7), the run's manifest and small evidence stay for longer, an UNHARVESTED run is never deleted by the tool (it is listed and reported), a run referenced by a plan item, an evidence README, a
release record or a recipe pin is protected, and the newest K runs per kind are kept regardless. Dry run is the default and prints what would be reclaimed (path, kind, size, reason, protection);
`--apply` deletes, touches nothing it cannot account for, and writes a cleanup record (path, size, date, where its summary lives), the same record the 2026-10-19 hand cleanup kept.

Disk pressure: the host reports usage; at 85% the tool warns and lists candidates, at 90% it applies the harvested-only policy automatically (configurable, off by default on shared hosts), and run
creation refuses to start below a free-space floor, naming the largest reclaimable items. It never runs while a measurement holds the host (the tree-activity lease).

## Steps

1. Policy file (per kind, in config/), manifest-driven candidate selection, protection rules, dry-run report.
2. `--apply` with the cleanup record and the lease check; tests with fixture work roots (protected, unharvested, expired, referenced, in use).
3. Disk-pressure hook in run creation and a scheduled invocation.

## Validation

Fixture tests for every protection; a dry run on Brutus after WSL06 lists the legacy runs without touching them; apply on a harvested test run removes only its bulk.
