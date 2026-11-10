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

## Notes

Source: gateway prior-work audit 2026-11-10 (missing item 1) and power review (defect 1). Related PWR04, RHP18.

## Change Log

- 2026-11-10T13:12:48.345318+00:00 (created-by): Created by agent
