---
id: PWR11
order: 0
plan: run-power-efficiency
state: deferred
created-at: '2026-11-25T01:08:44.064289+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# Whole-host deep idle and platform power modes

## Description

Identify safe reductions in unloaded and loaded-idle wall power across GPU runtime PM, PCIe links, NVMe, DRAM and peripherals. Board cap does not reduce idle (PWR10). PWR01 owns baseline, PWR03 CPU, PWR06 R9700 GFXOFF/BACO, BHK01 kernel recovery, BHK02/03 memory timings; do not duplicate them. Read-only feasibility first, then authorized host experiments.

## Steps

1. Measure wall and per-device unloaded, HIP-resident, loaded and post-request idle with PWR01; avoid monitor wakeups. 2. Inventory GPU PCI functions and bridges, runtime PM counters, links, open handles and monitoring references. Test resume/VRAM retention feasibility, distinguishing BACO/BAMACO/D3 with PWR06. 3. Read actual NVMe APST and supported states, inventory DDR5 power-down/self-refresh and supported BIOS settings using BHK02; identify NIC/peripheral losses. 4. Propose supported changes with maintenance, recovery and explicit host-owner approval; one variable at a time.

## Detailed Solution & Technical Design

No GPU board-power transaction or engine schema change. Host controls require host ownership, not a GPU lease alone. Keep owner-disabled ASPM unchanged until separate approval; never force it. No live memory register writes, BIOS/firmware changes or forced suspend of resident compute in feasibility stage. Reuse probe tooling; no new controller.

## Code Samples & Guidance



## Files



## Validation

Matched wall energy and sensor validity, runtime residency, repeated wake latency, workload/build quality and performance, storage/network stability, exact restore or supported maintenance rollback. Document negative/unsupported results.

## Effort & Risk



## Standards



## Acceptance Criteria

Reproducible read-only baseline and capability report; each authorized change has independent before/after wall energy and repeatable wake, workload/build stability and restoration evidence, or negative/unsupported disposition. No unapproved root writes or automatic activation. Cross-reference PWR01/PWR03/PWR06/BHK01–03/PWR10; no duplicate ownership.

## Notes

New distinct platform scope from 2026-11-25 power audit. Does not block first native minimum-cap profile.

2026-11-27 consolidation (deferred): Re-enter only if idle duty cycle and energy show material savings, with host authorization, wake/build/inference stability and restoration. No host writes now. All earlier results remain historical; see power-session-closeout inventory.

## Change Log

- 2026-11-25T01:08:44.064289+00:00 (created-by): Created by agent

## Reviews

- RV4691
- 2026-11-26T23:16:49.466715+00:00 (updated-by): Updated: section:notes
- 2026-11-26T23:19:40.754216+00:00 (state-transition): State: pending → deferred
