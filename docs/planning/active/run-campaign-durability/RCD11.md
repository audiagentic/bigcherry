---
id: RCD11
order: 11
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T01:20:22.304212+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Executor interface + LocalExecutor (Windows workstation GPU) and production measurement window

## Description

From the agreed design (docs/design/JOBS_ORCHESTRATOR.md): platform-neutral Executor (ResourceRequest/ExecutionRequest/ExecutionHandle/Allocation) with SlurmExecutor, LocalExecutor (Windows 11 workstation gfx1100 via HIP SDK, tests, emergencies) and FakeExecutor; execution_environment_hash in attempt identity (Windows HIP and Linux ROCm are separate series). Production coexistence: root-owned bigcherry-measure-window.service (RuntimeMaxSec=4h, idempotent exit, boot recovery) with sudoers limited to start/stop of that unit; GPU2/3 jobs use a 10 s production-idle attestation (llama-swap /running, backend /slots + /metrics); bigcherry host-run wrapper for ad-hoc builds/benches.

## Steps

1. jobs/executor.py interface + Fake; domain import guard test (no Slurm imports).
2. LocalExecutor on Windows with device lock and env-hash capture.
3. Measurement-window helper + units + sudoers (owner installs).
4. Idle attestation module + tests.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline tests; hardware: window overrun auto-closes and production /health recovers; a Windows-workstation job produces a separate-series record.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-26T01:20:22.304212+00:00 (created-by): Created by agent
