---
id: PWR09
order: 0
plan: run-power-efficiency
state: pending
created-at: '2026-11-10T13:32:42.998106+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# PWR09: transactional Python power controller and supervisor integration

## Description

Python library (tools/bigcherry/hardware/power_transaction.py, power_receipt.py) so the strict apply is implemented once, never in bash: whole-layout preflight against fresh capability probes, selected-BDF only, gpu lease ownership verified (launch.lock is not the lease), durable journal (tmp, fsync, rename, dir fsync) before the first write, deterministic BDF order, exact readback per setting, reverse-order rollback, restore after verified process exit, journal retained on failure, a stale journal blocks later explicit applies but not observe-only starts. Applies exactly the manifest's spec.plan.power (PWR08) and rejects profile/BDF/digest divergence; the final measured manifest's power plan must equal the applied one or the launch aborts and rolls back. Receipt (separate from the manifest) binds to metadata.content_digest and records start/post_apply/pre_restore/stop observations. Until RSC02 exists the shell launcher gets only an adapter that calls this library; an explicit --gpu-power failure aborts before engine start (today it continues, launch-flashnext-best.sh ~385-388). No automatic restore after SIGKILL or host crash is promised: durable recovery instructions only; a watchdog is a later slice (candidate for the Fable session).

## Steps

1. GPU capability probe (PWR07 PR2): per-BDF driver/hwmon by name and parent, OverDrive parsing with device-reported ranges, per-control capability, ambiguous hwmon refused. 2. Receipt types and atomic persistence. 3. Transaction coordinator with an injected fault layer. 4. Route existing gpu-power apply/restore through it (remove the whole-inventory path). 5. Launcher adapter, strict failure, run-specific cleanup. 6. Benchmark per-cell receipt refs (MEN11). 7. Read-only host smoke on this host. 8. Manifest consumption once PWR08 lands.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Fault-injection matrix: second-device preflight failure = zero writes; write rejected after preflight = rollback; readback mismatch = rollback with expected vs observed; rollback failure = journal kept, no success claim; stale journal; unselected GPU never changed; lease not verified = explicit writes refused, observe-only allowed; observe-only receipt IO failure does not stop serving. Host smoke makes no write call.

## Effort & Risk



## Standards

No legacy / no backward compatibility.

## Acceptance Criteria



## Notes

Steps 1-7 do not need v2 and proceed first. Fake sysfs proves portability behaviour, not that a real driver accepts a command. Parent of the remaining PWR04 work.

## Change Log

- 2026-11-10T13:32:42.998106+00:00 (created-by): Created by agent
