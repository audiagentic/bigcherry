---
id: RCD10
order: 10
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:23.945064+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Accept the v1 service, migrate active qualification work and retire the lab queue

## Description

Run the final v1 acceptance gate before removing the ad-hoc queue. Acceptance is capability/inventory-driven, not tied to today's slot numbers: cover every architecture/toolchain/model class needed by active qualification, at least one multi-GPU peer requirement when present, production-conflict and non-conflict paths, and the historical incident classes. Compare results with direct `validation_campaign` behavior and preserve rollback until the new path proves durable.

Archive/retire only queue/scheduling wrappers made obsolete. Do not remove lab analysis utilities until RCD08 replacements prove parity.

## Steps

1. Generate acceptance matrix from accepted RCD12 inventory + current active job specs; no hard-coded GPU indices.
2. Run offline full suite and lab planning simulator; require clean status.
3. Install/qualify RCD03 Slurm and RCD11 production gate.
4. Submit representative monolithic v1 jobs through `bigcherry jobs`, including active variations.
5. Force incident/recovery scenarios.
6. Compare evidence identity/verdict outputs with direct campaign invocation where safe.
7. Migrate queued work by creating canonical JobSpecs; do not import shell logs as completed service attempts.
8. Disable old queue entrypoints for a soak period while keeping documented rollback.
9. After acceptance/soak, archive shell queue/watch/switch wrappers; update docs/skills/references.
10. Produce an acceptance record with branch commit, Slurm config/inventory hashes, tests, hardware gates and known fallback mode.

## Detailed Solution & Technical Design

### Capability-derived matrix

At minimum instantiate jobs for each distinct active combination of:

- GPU architecture;
- Linux ROCm/toolchain identity;
- model class/VRAM threshold;
- single vs multi-GPU/peer access;
- producer class requiring special inputs/corpus;
- common patches/frozen validated composition;
- production lane enabled where used.

If an architecture is not physically present it cannot be accepted for Brutus and must be marked unsupported/pending hardware, not simulated as accepted.

Windows LocalExecutor acceptance is separate and never substitutes Linux series evidence.

### Incident matrix

Force safely:

1. harness exits 76 -> attempt fails; fix/new commit; retry same `run_id`, `attempt+1`;
2. transient exit 75 -> same attempt/commit requeue and restart count;
3. SSH/gateway client disconnect immediately after submit -> job continues and reconnect status/events work;
4. disable/re-enable queued run -> same series/session slot;
5. cancel running attempt -> process group gone/history retained;
6. root/work disk preflight failure and safe mid-run threshold breach;
7. stall fixture (not real timed scientific sample);
8. promotion modifies current validated recipe between sessions -> frozen series composition unchanged;
9. BigCherry branch advances while queued -> new attempt resolves latest only at attempt start;
10. production conflict -> exclusive window; no-conflict -> idle attestation/watchdog;
11. production config/process drift during sample -> sample invalidated as contamination, no scientific result;
12. hardware inventory drift fixture -> node drains/wakes and new work blocked;
13. measurement-window overrun -> automatic cleanup/production health recovery;
14. slurmctld restart -> queued ownership/status preserved;
15. cgroup acceptance or explicit fallback recorded.

Historical compiler/disk/parser/attestation/OOM cases should map to explicit FailureKind/exit behavior, never scientific FAIL.

### Migration

For each pending shell-queue item:

- parse the human source into a canonical JobSpec using a one-time migration tool/report;
- re-resolve contract/composition/model/toolchain identities;
- assign planned series/session explicitly;
- submit as new service work;
- retain old logs read-only under original lab location for provenance.

Do not infer success from `CAMPAIGN_EXIT=` old logs into new service state.

### Retirement

Candidates after soak:

```text
tools/lab/plan-qualification/queue.sh
tools/lab/plan-qualification/run_campaign.sh
tools/lab/plan-qualification/make-serial-2.sh
hand-written watcher/switch scripts
```

`work-root.sh` may be retired only after every supported caller uses core environment/work-root resolution. `summarize.py`/`noise.py` retire only after RCD08 parity.

Archive per repository TOOL_DISPOSITION rules rather than deleting history when appropriate.

### Soak/rollback

Minimum soak: complete one full planned series plus representative alternate-architecture/toolchain jobs with no manual scheduler intervention.

Rollback means stop submitting new service jobs and use direct campaign/manual lab path; never run two schedulers against the same resource set concurrently. Preserve new run-store records read-only.

## Code Samples & Guidance

Acceptance generator:

```python
def acceptance_matrix(
    inventory: HardwareInventory,
    active_specs: tuple[JobSpec, ...],
) -> tuple[AcceptanceCase, ...]: ...
```

It groups by scientific/resource capability, not device slot.

## Files

Planned:

- `tools/bigcherry/jobs/acceptance.py`
- `tools/tests/jobs/test_acceptance_matrix.py`
- `docs/reference/jobs/ACCEPTANCE.md`
- `docs/reference/jobs/MIGRATION.md`
- updates to tool disposition/reference docs
- old lab scripts archived only after gate.

## Validation

Offline prerequisite:

- all jobs/hardware/job-service tests green;
- acceptance matrix fixture changes correctly when cards are added/removed/swapped;
- migration parser refuses ambiguous legacy queue lines instead of guessing;
- mock planning simulator passes;
- repository search has no active docs instructing agents to use the old queue after retirement.

Hardware acceptance record contains:

- accepted inventory hash/device stable IDs/cohorts;
- installed Slurm version/config digest/cgroup mode;
- production config hash/policy;
- each acceptance run_id/attempt/evidence path;
- forced incident outcomes;
- evidence/report parity notes;
- explicit unsupported capabilities.

## Effort & Risk

Medium operational effort, high consequence. Main risk is premature script retirement; require soak and reversible cutover.

## Standards

No optional stopping; planned N remains fixed. Scientific result equality/parity is judged by existing evidence contracts, not scheduler exit codes.

## Acceptance Criteria

- all currently required Brutus capability classes have real acceptance or are explicitly unsupported;
- all incident scenarios recover/classify as designed;
- client/gateway lifetime does not own execution;
- production is protected in conflict/non-conflict paths;
- one full series completes without shell queue/watcher intervention;
- old queue wrappers are no longer referenced by active operational docs before archival.

## Notes

This is the v1 retirement gate. RCD07/RCD08 can remain later-phase unless their functionality is required to replace an operational script safely; `summarize.py`/`noise.py` therefore stay until RCD08.

## Change Log

- 2026-09-26T00:52:23.945064+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Replaced slot-specific acceptance with dynamic capability matrix, incident/soak/rollback and conditional tool retirement.
