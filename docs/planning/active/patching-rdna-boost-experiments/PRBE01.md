---
id: PRBE01
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:53:31.014682+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Implement final native-BF16 flash-attention logical experiment

## Description

Port and qualify the folded native-BF16 flash-attention logical path from the reviewed commit series. Existing isolated evidence is directionally promising (clean decode gain and no measured prompt cost) but correctness and cross-architecture repeat remain before promotion.

## Steps

1. Port only the reviewed logical BF16 path as one precision experiment; preserve F32 accumulation and native-BF16 guards.
2. Keep unsupported hardware on the existing fallback and preserve no-F16-regression behavior.
3. Separate cleanup/operand-order evidence from the implementation arm.
4. Run the FLASH_ATTN_EXT backend correctness matrix, BF16 same-path checks, and exact output/precision checks without claiming F16 bit identity.
5. Run long/deep-context numerical-quality checks and repeat on gfx1100 and gfx1201.
6. Reproduce isolated decode/prompt characterization with balanced interleaved controls, then gate promotion on correctness, precision, quality, fallback, cross-architecture, and performance evidence.

## Detailed Solution & Technical Design

Treat this as a precision-path experiment, not six cherry-picks. Validate architecture guards, unsupported fallback, F32 accumulation, BF16 operand semantics, graph capture, and numerical stability. Existing evidence: materialized patch 1202; isolated gfx1100 decode was directionally positive and prompt deltas were noise-adjacent/indistinguishable at pp>=1024. This is not a promotion claim.

## Code Samples & Guidance



## Files

Future vendor HIP flash-attention sources; patches/1202_rd04_bf16_flash_attn_tile; correctness fixtures and backend matrix; balanced E2E evidence for gfx1100/gfx1201

## Validation

Build and apply/idempotence; FLASH_ATTN_EXT backend matrix; BF16 same-path correctness; F32 accumulation; long/deep-context numerical quality; gfx1100/gfx1201; unsupported fallback; graph capture; no F16 regression; balanced decode/prompt performance with repeatability.

## Effort & Risk



## Standards

Correctness before performance; explicit precision contract; fallback preservation; no claim of F16 bit identity.

## Acceptance Criteria

All RD04 requirements are either passed with evidence or explicitly dispositioned: F32 accumulation/native-BF16 guards, unsupported fallback, FLASH_ATTN_EXT correctness matrix, BF16 same-path checks, long/deep-context quality, gfx1100/gfx1201 coverage, unsupported fallback, no F16 regression, and balanced performance. No promotion on isolated throughput alone.

## Notes

Supersedes: RD04
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd04

Supersedes: RD04
Inherited semantic scope: preserve every actionable RD04 gate; historical evidence remains on completed predecessor.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T10:53:31.014682+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:10:07.152011+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.129465+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.797954+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:10:06.643778+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_021313_the-semantic-audit-is-now-trac_4827
- 2026-09-10T02:13:13.308366+00:00 (updated-by): Updated: section:ledger-events
