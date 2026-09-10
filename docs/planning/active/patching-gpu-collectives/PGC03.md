---
id: PGC03
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-09T10:48:00.988302+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P1
---

# Brutus kernel PCIe P2P DMA quirk for dual-XTX: patch, build, and boot on ROCm-validated 7.0.13 line

## Description

Complete production disposition and newer-ROCm validation for the narrowly-scoped Brutus dual-XTX PCIe P2P DMA quirk. The kernel remains diagnostic until promotion criteria are met.

## Steps

- Keep the quirk scoped to the exact XTX endpoint IDs/BDFs and CPU PEG root-port pair; preserve fallback kernel and visible boot-menu recovery.
- Validate on the ROCm-validated 7.0.13 line with clean build/install/boot and post-boot rocm-smi/dmesg checks.
- Verify all 12 directed GPU pairs: only XTX 0<->1 may report P2P; all R9700/6900XT pairs remain disabled.
- Use correctness-checked push-direction transfers and RCCL debug/topology evidence; do not rely on the retracted pull benchmark or raw bandwidth alone.
- Decide diagnostic-only vs production promotion with a non-dirty package/version, rollback path, and rerun GP11 native/BC comparisons only if RCCL actually selects the path.

## Detailed Solution & Technical Design

The quirk is a platform/endpoint/topology admission change, not proof that kernel-side peer reads or AllReduce improve. Preserve the corrected finding: push DMA has modest small/medium-transfer benefit, while GP11's corrected internal AllReduce found host staging can win. Production promotion requires RCCL transport evidence and no regressions.

## Code Samples & Guidance



## Files

Brutus linux v7.0.13 quirk tree/package; fallback boot entry; P2P pair matrix; correctness-validated push benchmark; RCCL topology logs; GP11 rerun artifacts and rollback record.

## Validation

Kernel build/checkpatch/boot; rocm-smi/dmesg; all 12 peer pairs; per-element push correctness; RCCL transport selection; topology-specific latency/bandwidth; native/BC comparison only with validated transport.

## Effort & Risk



## Standards

Exact endpoint/root-port scoping; fail-safe fallback kernel; corrected benchmark provenance; never use retracted pull results.

## Acceptance Criteria

Promotion occurs only with exact scoped quirk, clean validated boot, no unintended peer access, RCCL transport confirmation and a measured production benefit without correctness regression; otherwise retain diagnostic-only status.

## Notes

Supersedes: GP13
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-gpu-collectives-gp13

## Change Log

- 2026-09-09T10:48:00.988302+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:04.729926+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.780752+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.245031+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:38:11.686425+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023824_the-gpu-collective-successors_5773
- 2026-09-10T02:38:24.281852+00:00 (updated-by): Updated: section:ledger-events
