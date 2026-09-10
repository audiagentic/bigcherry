---
id: PRBE60
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:41.013607+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: null
---

# SYS-PCIE-001: PCIe ASPM performance policy

## Description

Document and optionally validate Linux PCIe ASPM performance policy as host tuning for bandwidth-bound discrete RDNA systems; never implement a runtime toggle.

## Steps

Recheck R9700 discussion evidence; compare default versus performance ASPM policy on R9700/XTX with dense and MoE workloads, single-GPU fully resident and transfer-heavy controls, cold boots, PCIe link state, TG/PP, transaction latency, and idle/power; document only locally reproduced results.

## Detailed Solution & Technical Design

This is OS-level operations guidance outside llama.cpp. Aggressive link power saving may increase transfer latency on some systems but is topology/system dependent. Do not bake it into runtime or claim a universal setting.

## Code Samples & Guidance



## Files

Host tuning documentation and reproducible measurement record; no llama.cpp runtime patch.

## Validation

No output impact; TG/PP, PCIe latency, power/idle and link-state evidence across cold boots/workload controls.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Document as optional host tuning only if local hardware reproduces a repeatable benefit; never add a runtime ASPM toggle or universal recommendation.

## Notes

Supersedes: RD77
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd77

## Change Log

- 2026-09-09T10:57:41.013607+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:51.599098+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.398077+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.212166+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:15:55.582608+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031644_repaired-four-more-active-succ_8062
- 2026-09-10T03:16:44.945234+00:00 (updated-by): Updated: section:ledger-events
