---
id: RSC16
order: 0
plan: run-serving-configuration
state: pending
created-at: '2026-11-27T03:33:31.874138+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Generic shared KV capacity independent of request context and concurrency

## Description

Deliver model-neutral shared paged cache planning and admission for independent request length, session slots and total capacity. Client harness owns individual limits. Design: docs/design/radiance-shared-kv-capacity.md. Do not assume S*L resident. Preserve legacy behavior and physical safety.

## Steps

1. RSC17 owns exact native startup proposal, live admission, retention and versioned facts; RSC18 owns verification-first runtime and targeted corrections. 2. Integrate on final merged bump baseline; keep bump separate. Reuse affected patch-chain and existing harness gates; one integrated canonical candidate build after host gates. 3. Managed installed proof of actual capacity and active mixed long/short progress, legacy/supported-family and existing quality/performance acceptance. 4. Inventory reachable callers and retire redundant paths only after parity and preserved evidence; record follow-up triggers, no new plan owners.

## Detailed Solution & Technical Design

Code guide I5 owns controlled retirement, cross-family acceptance and conditional follow-ups; I1-I3 delegated to RSC17, I4 to RSC18. No separate cleanup plan or reopening historical completed work. Maintain existing acceptance and reuse ENG04/ENG03/MEN17.

## Code Samples & Guidance

Use managed patch metadata, composition, caller inventory and archive lifecycle; verify legacy replay/fixtures and installed qualification before retiring redundant split or refit. Guide I5 documents boundaries and measured triggers for optional work.

## Files



## Validation

RSC17 and RSC18 card-free gates required. RSC16 owns managed installed GPU proof, actual per-rank/group capacity versus observed pages and allocations, mixed resident lengths, accuracy and performance, legacy and unsupported-path regressions. Owner-approved thresholds before promotion.

Guide I5: review scope and actual caller/link inventory, legacy fixture/replay checks, patch-lint and affected chains, staged installed qualification and preserved historical evidence. Existing ENG04/03 and MEN17 evidence reused.

Require target production chain identity, preallocation proposal + live verifier, actual configured carve, late startup/front-ready, and simultaneous active/resident long decode with short progress. Unique prompts first, sharing control second; physical refusal not target pass.

## Effort & Risk



## Standards



## Acceptance Criteria

No S*L backing demand in shared mode; independently reported L/S/C; measured per-rank and tier capacity; at least one L request or actionable refusal; mixed-length requests bounded by actual pages and slots, no deadlocks or lost live state; legacy parity and supported-family matrix before rollout.

No duplicate plan owners, no premature patch or evidence deletion, no unqualified behavior on unsupported paths; record explicit retirement or retention decision for every cleanup candidate.

## Notes

Motivation: 2020.23.1 IQ8R TP3, yarn:2, bf16_i8, 524288, 16/8/4 slots and batch512 refused RAD_E_NOMEM at refit; no inference. RSC12 owns variants, ENG04 owns format/quality, MEN17 release acceptance. No GPU or code changes in planning stage.

Code-level review RV4697: reuse existing fresh-prefill/recompute guards and KVManager; measure rectangular BatchBuilder overhead before changing ABI; replace fixed-inclusive kv_one/L assumptions in shared mode; rank feasibility must use same-rank costs. See docs/design/radiance-shared-kv-capacity-code-guide.md. No implementation yet.

RV4698 second review: rad0029 already solves a subset of per-rank capacity; rad0011 common gate runs earlier. Revised design reuses it, avoids new scheduler where not needed, corrects first-test profile mismatch and demands actual calibration and physical ledger. Scope first shared-mode qualification to supported dense full-paged discrete paths, extend capability-driven across models/tiers only with gates. See guide H. No code or GPU changes.

RV4699 third review: consolidated all cleanup into existing three plans. Preserve RSC12/RHP16/20 in-progress ownership, RSC15 deferred scope, ENG07/RAD38 format work and PVPS14 checkpoint investigation. No engine changes.

RV4700 fourth deep review, guide J: exact page vector guarantee and cap semantics; source inspection used experimental slice, archived .1 production slice differs. Do not promote plan-only measurement to actual capacity. Initial qualification fixed-backed single text engine including hybrid state/window geometry; fleet flex legacy until separately proven.

Guide K is normative 2.3 implementation map, A-J requirements retained. Source compare confirmed native improvements, all 34 recorded base-chain hashes matched reviewed bump controller; final recipe/source identity still required. No new GPU kernels or scheduler for shared capacity. Bump syntax qualification is not RSC16 acceptance. Reuse one canonical candidate build and existing ENG03/04/MEN17 evidence; no repeated engine builds per host change.

## Change Log

- 2026-11-27T03:33:31.874138+00:00 (created-by): Created by agent

## Reviews

- RV4697
- 2026-11-27T03:42:47.951296+00:00 (updated-by): Updated: section:steps, section:notes, section:validation
- RV4698
- 2026-11-27T03:48:04.634120+00:00 (updated-by): Updated: section:notes
- RV4699
- 2026-11-27T04:01:00.468365+00:00 (updated-by): Updated: section:detailed_solution, section:code_samples, section:notes
- 2026-11-27T04:01:33.215760+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria
- RV4700
- 2026-11-27T04:18:54.576854+00:00 (updated-by): Updated: section:notes, section:validation
- 2026-11-27T07:13:29.577610+00:00 (updated-by): Updated: section:steps, section:notes
- RV4750
