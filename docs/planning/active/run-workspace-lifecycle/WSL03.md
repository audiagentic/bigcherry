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

## Execution gate review (2026-10-23)

- The current `core/quarantine.py` implements verified-harvest file inventory, same-filesystem journaled rename, restore and purge. This is not evidence that host-wide automatic cleanup is safe.
- **WSL03-A:** fail-closed active-work and lease coordination per physical work root, shared across checkouts; identify process start time and namespace to avoid PID reuse. Handle corrupt/missing lease data as protected. Establish all active producer coverage before apply.
- **WSL03-B:** enforce host opt-in, typed GC roots, minimum age 7 days, >=48-hour quarantine grace, per-pass byte cap, no symlink traversal/cross-device moves, append-only cleanup journal and resumable restore. Recheck exact evidence bundle and candidate identity at mutation time; never broaden from verified bulk to entire runs.
- **WSL03-C:** offer automated disk-pressure warning/read-only inventory first. Automatic deletion stays disabled until shared locks/leases, producer participation, adoption and at least two dry-run cycles are proven. Quarantine does not reclaim space; purge does.
- Regression tests: fault injection between each journal and rename step, active/foreign/unknown paths, hard links, changing bundles, stale lock, race between selection and apply, interrupted restore, multiple concurrent checkouts and same filesystem aliases. No destructive test on live unharvested runs.

## Scope (owner, 2026-10-24)

`bigcherry clean` cleans the workspace it runs in (its worktree, runs and builds), not a host-wide sweep. Worktree
hygiene is `bigcherry slice finish` after merge and `bigcherry slice prune` at session end. Host-wide automatic
deletion and the WSL06 legacy adoption are out of scope.

## Validation

Fixture tests for every protection; a dry run on dev after WSL06 lists the legacy runs without touching them; apply on a harvested test run removes only its bulk.

## Design review (GPT, 2026-10-19) - adopted

- GC roots are typed references (a run naming a run/build/source/object), active runs and leases, release records, administrative holds and the newest K per kind. A plan id in a manifest is metadata and pins
  nothing; an evidence README is explanatory, not authoritative (a machine-readable `evidence.index.json` naming run ids, bundle ids and hashes is the root).
- Unmanifested, unreadable, unknown-kind, unknown-owner or foreign directories are protected and reported, never candidates.
- Order of enabling: report disk pressure first; deletion only through quarantine (rename into trash/, TTL at least 48 h, then a bounded purge) after observed dry-run cycles; automatic deletion at a pressure
  threshold stays off until ownership and leases are proven on the host. Guards: per-pass byte limit, minimum age (7 days to start), no cross-filesystem moves, never follow symlinks, append-only cleanup journal,
  global maintenance lock, explicit enablement per host. Quarantine does not reclaim space; purge does.
- Lease limits to fix before apply: leases are scoped to project_root (need a shared namespace per physical work root), a PID check is vulnerable to reuse, malformed lease files must block cleanup, and every runner must
  take a lease (an existing one that does not is unsafe).
- Done in the first slice (PR #311): the read-only inventory and dry-run plan with every protection above that does not need harvest or leases.

Revised order: WSL01 foundation, WSL03 dry-run skeleton, WSL02 harvest, WSL03 quarantine/apply, WSL05 templates and producer migration, WSL04 build/source GC, WSL06 adoption, disk-pressure automation.
