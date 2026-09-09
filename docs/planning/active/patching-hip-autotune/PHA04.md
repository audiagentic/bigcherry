---
id: PHA04
order: 0
plan: patching-hip-autotune
state: pending
created-at: '2026-09-09T10:48:57.946592+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# RCCL admission gate: implement level-2 per-call candidate binding

## Description

Assessment complete: the current 0830 per-call seam exposes requested/effective provider and reduction signature telemetry, but not the RCCL algorithm/protocol/channel candidate identity required by level-2 admission. PHA04 therefore depends on a settled PHA03 topology identity and an upstream/RCCL observation seam that can prove the actual candidate; no candidate allowlist or silent fallback is being invented prematurely.

## Steps

1. Settle the PHA03 topology identity/admission contract and preserve its fail-closed ordering.\n2. Add an observation-only candidate identity at the real RCCL per-call selection boundary, including algorithm, protocol, channels, exact compatibility/build identity, reduction signature, and topology identity.\n3. Encode qualified versus unsupported versus performance-excluded outcomes from THA07 without conflating safety and performance.\n4. Bind the gate immediately before RCCL dispatch; unknown or unqualified candidates must fail closed.\n5. Add synthetic and real Brutus validation before any default-build promotion.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patches/0830_split_reduce_telemetry/patch.py; patches/1225_hi85_nccl_heterogeneous_arch_guard/patch.py; tools/bigcherry/profiling/rccl_schema.py; tools/bigcherry/profiling/rccl_qualify.py; docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md

## Validation

Current source review confirms `ggml_hip_reduce_telemetry_provider()` carries provider/signature/topology observations only; it does not carry RCCL algorithm/protocol/channel identity. Existing RQ08/RQ10 evidence is scoped to its original artifacts and cannot be promoted into a per-call gate without that identity. Implementation remains intentionally pending PHA03 and the required observation seam.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

A level-2 gate must admit only an exactly qualified RCCL candidate for the current compatibility/build, reduction signature, topology identity, and candidate tuple; it must fail closed for unknown/unqualified candidates and preserve an explicit performance-exclusion distinction. No default-build promotion occurs without real hardware evidence.

## Notes

Supersedes: HI147
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-hip-autotune-hi147

## Change Log

- 2026-09-09T10:48:57.946592+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:05:02.519090+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.848546+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T13:42:25.760165+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:acceptance_criteria
- chg_20260909_134238_pha04s-dependency-and-evidenc_1280
- 2026-09-09T13:42:38.013204+00:00 (updated-by): Updated: section:ledger-events
