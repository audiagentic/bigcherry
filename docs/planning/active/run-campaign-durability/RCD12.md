---
id: RCD12
order: 12
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T01:39:24.166357+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Dynamic GPU hardware discovery, generated architecture-typed GRES, drift drain, hardware-cohort series identity

## Description

Round-4 design (docs/design/JOBS_ORCHESTRATOR.md section 9): tools/bigcherry/hardware/ discovery (AMD-SMI UUID > RSMI unique_id > serial > weak; BDF/renderD/index are observations), bigcherry-hardware-discovery.service before slurmd with observed/accepted inventory and generated gres.conf typed by architecture, drain + wake on drift, GpuRequirement capability requests with over-allocation for exact/subset constraints (ROCR_VISIBLE_DEVICES by UUID), series keyed by platform_environment_hash + hardware_cohort_hash, environment.local.toml reduced to host policy + aliases, Windows HIP probe discovery.

## Steps

1. Probe real UUID/unique_id availability on 2x gfx1100, gfx1201, gfx1030 (amd-smi / rocm-smi).
2. hardware package + inventory hash + topology fingerprint.
3. gres.conf/node generation + drift state machine.
4. GpuRequirement resolver + UUID selection.
5. Series identity migration.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline tests with recorded inventories (add/remove/swap/move); hardware: unplug-free simulation via recorded snapshot drift drains the node; generated gres.conf passes slurmd -G.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-26T01:39:24.166357+00:00 (created-by): Created by agent
