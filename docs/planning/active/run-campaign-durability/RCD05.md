---
id: RCD05
order: 5
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:05.135167+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Implement attempt resolution, pinned workspaces and monitored execution

## Status — 2026-09-28

**Software execution path implemented; production host qualification and retention remain.**

Implemented:

- exact commit resolution per attempt and detached attempt workspace;
- scientific-identity recheck at attempt start; drift blocks instead of mutating a series;
- typed `validation_campaign` argv with allocation-local `--device-map` only;
- local stable-device OS locks vs external Slurm resource ownership;
- pre-spawn accepted-inventory/allocation stable-ID attestation;
- durable `executor-start.json`, `allocation-attestation.json`, `executor-result.json`;
- file-backed `experiment.bundle.run_managed()` streaming; Python memory does not scale with child output;
- process-group TERM -> bounded grace -> KILL support;
- attempt runner disk guards for project/work filesystems, with absolute and fractional free-space floors;
- CPU/output liveness stall detector;
- monitored validation child execution with typed `disk_pressure`, `stalled`, `launch_error`, `hardware_attestation` and configuration incidents;
- retryable environment incidents return 75; harness/configuration incidents return 76; scientific FAIL remains normal campaign completion.

Default disk floors are host-overridable environment policy:

```text
BIGCHERRY_ROOT_MIN_FREE_GIB=50
BIGCHERRY_WORK_MIN_FREE_GIB=200
BIGCHERRY_ROOT_MIN_FREE_FRACTION=0.05
BIGCHERRY_WORK_MIN_FREE_FRACTION=0.02
BIGCHERRY_MONITOR_POLL_SECONDS=15
BIGCHERRY_STALL_SECONDS=0
BIGCHERRY_TERM_GRACE_SECONDS=5
```

`BIGCHERRY_STALL_SECONDS=0` intentionally disables automatic stall termination until a host-specific nonzero threshold is qualified; CPU/output progress detection is implemented and tested.

## Remaining work

- choose/qualify Brutus nonzero stall threshold;
- implement retention/cleanup policy (never delete active/unharvested evidence; only eligible harvested/failed work roots);
- Brutus acceptance for real disk pressure, process-tree cancellation, very large server logs and gitignored vendor/environment workspace inputs;
- Windows-local process-tree/cancellation acceptance.

## Invariants

- run/session identity survives harness retries; attempt identity does not;
- retry always creates a new attempt/native job in monolithic v1;
- no stash/canonical-checkout mutation is required;
- launch selectors are local/transient; stable GPU IDs are scientific identity;
- infrastructure incidents never become scientific performance FAIL.

## Change log

- 2026-09-27: pinned workspace/runner/identity/allocation-local mapping implemented.
- 2026-09-28: reconciled streaming and stable-ID attestation; wired typed disk/stall process monitoring into the real attempt runner.
