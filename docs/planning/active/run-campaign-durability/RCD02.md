---
id: RCD02
order: 2
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:51:52.243021+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# Job service platform decision: Slurm single-node vs pueue vs custom daemon (adversarial GPT review to agreement)

## Description

Decide the durable job-processing platform for campaign jobs on Brutus. Leading option: host-installed Slurm (GPU gres by device, licenses for host-exclusive timed measurement and build slots, arrays, dependencies, requeue, sacct) with a thin BigCherry layer; alternatives pueue or the custom SQLite daemon from docs/design/JOBS_ORCHESTRATOR.md. Not containerized (BRVP02 covers toolchain images).

## Steps

1. GPT round on build-vs-buy, service shape, consolidation, observability (session ses_4019cfc56e774dd2).
2. Adversarial rounds against the 15 variations and incident list until no open objections.
3. Record the agreed design in docs/design/JOBS_ORCHESTRATOR.md; update RCD03-RCD10 to match.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Design doc states the decision, rejected alternatives with reasons, and an explicit 'agreed, no open objections' GPT turn.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner leaning Slurm (2026-09-26).

## Change Log

- 2026-09-26T00:51:52.243021+00:00 (created-by): Created by agent
