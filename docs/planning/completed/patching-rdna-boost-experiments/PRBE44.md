---
id: PRBE44
order: 0
plan: patching-rdna-boost-experiments
state: superseded
created-at: '2026-09-09T10:56:32.505332+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-GDN-003: Native exp2 decay in chunked GDN

## Description

SUBSUMED-INTO-PRBE42 (disposition: superseded, not a separate implementation). Patch 1221_rd50_gdn_chunked_recurrence's SUMMARY.md states RD52 (native exp2 decay, this item) is an inline micro-decision inside the same chunked-kernel body as RD50, folded into 1221 rather than a separate patch. No standalone RD52 code path exists to qualify; exp2 decay correctness/tolerance is part of 1221's overall correctness campaign (tracked under PRBE42), specifically the long-sequence recurrent-state/logit/PPL tests already specified there.

## Steps

1. Recheck source #54 and PRBE42's decay/state representation. 2. Replace the heavier expf/OCML decay calculation with __builtin_amdgcn_exp2f or equivalent native base-2 formulation only under the exact PRBE42 hardware/shape gate. 3. Define conversion/scaling and edge-decay handling so state evolution remains within the approved tolerance. 4. Add long-sequence state/logit/PPL tests, sensitive short-sequence and edge-decay controls, and architecture fallback tests. 5. Profile instruction mix and benchmark GDN and end-to-end prefill against PRBE42 baseline.

## Detailed Solution & Technical Design

Use native AMD exp2 in the chunked GDN decay calculation to reduce scalar math overhead. Convert the decay exponent consistently with the source formulation, preserve finite/edge handling, and leave the existing expf path for unsupported architectures or numerically sensitive configurations. This is an optimization within PRBE42, not a replacement for its recurrence eligibility gate.

## Code Samples & Guidance

Trigger: long prefill sequences with the PRBE42 supported decay range. Controls: short numerically sensitive sequences, edge decay values, non-RDNA targets, and PRBE42 baseline. Boundary: input decay distribution and context length.

## Files

PRBE42 GDN decay implementation and architecture selector; state/logit tolerance and PPL tests; edge-decay/short-sequence controls; instruction-profile and PP replay evidence for AMD-GDN-003.

## Validation

Correctness: recurrent state and logits within defined tolerance against PRBE42 baseline over long sequences, with PPL/deterministic checks and edge-decay controls. Performance: instruction profile, GDN kernel time, and PP E2E with variance. Acceptance: enable only if quality remains within tolerance and kernel/E2E benefit survives; otherwise retain expf fallback.

## Effort & Risk

M; approximate exp2 can accumulate recurrent error over long contexts. Keep tolerance explicit, test edge cases, and fail closed.

## Standards

Preserve PRBE42 hardware/shape gating, recurrent-state quality, unsupported-architecture fallback, and evidence provenance.

## Acceptance Criteria

Acceptance requires long-sequence recurrent state/logit quality within defined tolerance, edge-decay and short-sequence safety, supported-architecture fallback, and a repeatable kernel or E2E prefill benefit; otherwise retain the original decay path.

## Notes

Supersedes: RD52
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd52

Supersedes RD52. Depends on PRBE42; no standalone validity outside the chunked GDN kernel.

append

2026-09-24 relevance at b11126: SUPERSEDED/subsumed, same evidence as PRBE43 (patches/1221_rd50_gdn_chunked_recurrence/SUMMARY.md explicit subsumption statement for RD51/RD52/RD53). PRBE42's validation plan (long-sequence recurrent-state/logit/PPL parity) already covers whatever decay-math approach 1221 actually uses; no separate qualification runner needed. No GPT session needed.

2026-09-24 GPT review req_c18183e0a9034c94: READY

## Change Log

- 2026-09-09T10:56:32.505332+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:45.997534+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.328390+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.097809+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:07:07.712309+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-10T03:07:27.823252+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030747_carried-forward-the-remaining_4294
- 2026-09-10T03:07:47.473759+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:34:32.076326+00:00 (updated-by): Updated: section:description, section:notes
- 2026-09-24T02:34:55.912391+00:00 (updated-by): Updated: section:notes
- 2026-09-24T02:35:10.335523+00:00 (state-transition): State: pending → superseded
- chg_20260924_023553_re-scoped-11-rdna-boost-planni_1625
- 2026-09-24T02:36:37.279386+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:33:57.280153+00:00 (updated-by): Updated: section:notes
