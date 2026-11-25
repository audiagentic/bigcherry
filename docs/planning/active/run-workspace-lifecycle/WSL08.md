---
id: WSL08
order: 0
plan: run-workspace-lifecycle
state: pending
created-at: '2026-11-10T13:12:48.345318+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# The gpu lease covers the lifetime of the serving processes a launcher detaches

## Description

launch-flashnext-best.sh starts the engine and helpers detached from the command that holds the lease. Run inside `agent-lease acquire gpu -- bash body.sh` (runs/policy-lag, 2026-11-10) the lease tool still lists them as 'gpu processes outside any lease' (r3_moe_worker x3, radiance), and the lease releases when the wrapped command ends, so a launcher that returns while servers keep running leaves them unleased. Other agents share the cards (the VTP agents hold the lease in turns) and the same gap blocks safe power-state ownership (PWR04) and unattended campaigns.

## Steps

1. Decide how a lease owns a process tree that detaches (a supervised start under the lease, a lease-held-until-stop mode, or process-group/cgroup tagging) and what happens when a lease expires with a server alive.
2. Make the launcher start and stop under the lease for the whole serving life, and make `agent-lease status` list the servers as leased.
3. Test: two launches contend; a launcher is killed while servers run; the lease expires with a live server.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

agent-lease status lists launcher-started servers as leased for the run; a second acquire refuses while they live; a killed launcher leaves a recoverable state (no orphan holding the cards unleased).

## Effort & Risk



## Standards



## Acceptance Criteria

No GPU process started by the launcher is ever reported outside a lease, and no other agent can acquire the lease while one runs.

Power-specific cases: detached start cannot inherit a short-lived wrapper lease and later mutate; two owners contend, owner killed while children exist, expiry while serving, restore only after independently verified exit with authorized recovery; stale journal blocks new writes. Foreground stop/restore before lease release remains intact. Cross-link RSC02/PWR08.

## Notes

Source: gateway prior-work audit 2026-11-10 (missing item 1) and power review (defect 1). Related PWR04, RHP18.

2026-11-25 native power recovery dependency: RSC02 owns verified child-exit restoration, this item owns detached start lease lifetime, expiry and crash ownership. Foreground bc-serve serve in verified lease is first supported hardware apply path. Do not enable detached power application until lease persists through owner/children, verified stop/restore, and expiry/kill recovery with new mutation fenced; no promise SIGKILL auto restore without real proof.

2026-11-25 owner direction: no launch manifest, all authored serving configuration YAML, no compatibility shims or legacy backports. docs/design/radiance-native-yaml-launch.md supersedes earlier manifest-launch instructions; completed RUM04/05/06/09/12 evidence is historical, not proof of the new launch path. Existing capture manifests may remain output-only. No new programme or second resolver. Preserve foreground lease through all measurement/helpers/native Fleet and exact restore. Detached owner lifetime and stale-journal/crash/expiry proof remain here; no authorization via capture manifest. RSC02 owns transaction, no second supervisor.

## Status 2026-11-19

The launcher that detached serving processes is retired (S16 #1131); `bc-serve serve` is a foreground owner and `agent-lease acquire gpu ... -- <command>` holds the lease for exactly the owner's lifetime (used for every hardware run on 2026-11-19: the lease was held until the owner had stopped its children and restored power, and was free afterwards). What remains of this item is the case of a start that is left running on purpose (`bc-serve start` detached): decide whether the lease should then belong to the owner process, with `bc-serve stop` releasing it.

## Change Log

- 2026-11-10T13:12:48.345318+00:00 (created-by): Created by agent
- 2026-11-19T07:38:29.670825+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4585
- 2026-11-25T01:06:41.047513+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- RV4599
- 2026-11-25T01:41:03.020836+00:00 (updated-by): Updated: section:notes
