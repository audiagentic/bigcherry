---
id: THA08
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:48:50.169555+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Investigate whether the HI141 w4:nw8:rpb1:sk0 defect corrupts shared MTP trunk state (draft AND target)

## Description

Determine whether the HI141 w4:nw8:rpb1:sk0 candidate's numerical defect is amplified by coupled target/draft MTP contexts.

## Steps

Map Qwen3.8-27B MTP target ctx_tgt versus separate draft ctx_dft graph and hidden-state handoff; inspect which context emits signature ne1=[5120,4,1,1]; determine shared-trunk versus head-specific operation; correlate candidate routing/defect with acceptance collapse and inform HI143 corpus/gate design.

## Detailed Solution & Technical Design

Correct the mental model: target and draft contexts execute separately, but target hidden states feed the draft MTP head. A global signature-keyed cache can route identical signatures in either context to the same candidate, so test correlation through the actual handoff rather than assuming shared trunk execution.

## Code Samples & Guidance



## Files

MTP graph construction/server speculative path; dispatch signature traces per context; HI141 candidate evidence; HI143 behavioral-gate inputs.

## Validation

Real MTP traces identify context/producer, candidate parity and acceptance behavior; independent/head-specific result determines whether correlation hypothesis survives.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close with an evidence-backed shared/independent classification and updated HI141/HI143 gate; do not infer causality from signature equality alone.

## Notes

Supersedes: HI144
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi144

## Change Log

- 2026-09-09T10:48:50.169555+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:54.088634+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.838080+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.342125+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:27:21.489000+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032737_repaired-four-major-tuning-suc_7897
- 2026-09-10T03:27:37.926185+00:00 (updated-by): Updated: section:ledger-events
