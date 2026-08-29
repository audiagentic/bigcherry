---
id: HTR04
order: 0
plan: hip-tune-recovery
state: pending
created-at: '2026-08-29T13:47:23.898965+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
---

# Retune escalation: recommendation-only signal, never an autonomous action (deferred implementation)

## Description

Adversarially designed with GPT (session ses_330ae3c055084f38, 2026-08-29), following on from HTR01. HTR01 deliberately never retunes -- it only reselects among already-measured alternatives or falls back to native. This item is about the DIFFERENT, harder question: when has recovery search exhausted what it can do, such that spending real GPU time on a genuine retune (or widening a signature's candidate search space) might be worth it?

GPT's central correction, adopted as this item's governing principle: 'HTR01 failure != retune.' The correct chain is HTR01 failure -> native fallback -> quantify the real loss -> IF material -> determine whether it's stale measurements or a genuinely exhausted search space -> only THEN an explicit retune recommendation. Never 'HTR01 failure -> retune' directly -- that risks the system repeatedly spending real GPU time remeasuring the exact same candidate pool it already knows it cannot safely deploy.

## Steps

1. (Already landed alongside HTR01, not blocked on this item) recovery.py emits RetuneRecommendation records with reason='alternatives_exhausted' whenever a signature falls back to native despite having had real measured alternatives -- pure structured evidence, asserted to change NOTHING automatically.
2. This item's actual scope is the three reasons GPT specified that HTR01's own data cannot yet support, and are each real, separate pieces of work:
   - `search_space_expansion_available`: requires knowing whether the candidate registry/manifest that produced a signature's measured alternatives has since grown (new candidate kernel variants added) that were never measured for that signature -- a registry-diff capability that does not exist today.
   - `repeated_behavioral_implication`: requires HTR02's BehavioralFailureWitness history to exist and be queryable across campaigns first -- explicitly sequenced AFTER HTR02, not parallel to it. GPT's explicit caveat: repeated implication under DIFFERENT candidate ensembles does NOT itself prove a candidate family is intrinsically bad (non-monotonic masking) -- this may only ever raise search-expansion PRIORITY, never trigger anything by itself.
   - `material_native_fallback_loss`: requires a real economic-loss estimate, and GPT was explicit this must NOT be naive `winner_effective_us - native_effective_us` for one op -- it needs to incorporate real dispatch frequency and, ideally, an actual E2E benchmark delta (a 30% kernel win on an operation contributing 0.1% of runtime is not worth a sweep). This is real new measurement/estimation work, not a lookup.
3. v1 remains recommendation-only, full stop -- GPT explicit: 'No autonomous GPU spend.' The RetuneRecommendation schema (already emitted, see recovery.py) is the complete v1 deliverable for THIS reason; an operator reads workdir/recovery-result.json and decides manually whether to launch a targeted retune. Automatic retuning is explicitly named as a MUCH LATER possibility requiring 'an explicit GPU-budget policy' that does not exist and is not designed here.
4. Before designing 'targeted single-signature retune' as an HTR01-adjacent mechanism (an idea raised and explicitly rejected as premature by GPT in this same negotiation): GPT identified that BigCherry's existing tune_promotion pipeline applies an EXPERIMENT-WIDE Benjamini-Hochberg statistical correction, so casually splicing a narrow single-signature remeasurement into an existing promoted.jsonl is NOT safe -- 'new narrow measurements + old promoted.jsonl => casually splice winner' is explicitly listed as a DO NOT. A real targeted-retune design needs its own explicit semantics for measurement/source/build/hardware/runtime identity, a fresh paired native reference, a noise/repeatability canary, candidate-registry/search-space identity, how the new hypothesis participates in the existing BH family, fresh correctness evidence, and the final full-cache HI143 gate -- GPT's verdict: 'until those semantics exist, targeted retune is a new measurement transaction, not an HTR01 extension.' Do not attempt to design this until HTR01 has real-hardware usage history (see notes).

## Detailed Solution & Technical Design

GPT's stated reason for deferring implementation (adopt verbatim as this item's gate before any further design work): 'HTR01 has not yet established: how often recovery exhausts alternatives; how often native fallback is materially costly; typical number of implicated signatures; whether candidate-family exhaustion is common.' The sequencing GPT specified: make HTR01 expose the structured escalation data (DONE, landed with HTR01's own commit) -> create this deferred item (DONE, this item) -> real-hardware validate HTR01 on several real campaign failures -> only THEN design targeted-retune transaction semantics from actual observed evidence, not speculation.

## Code Samples & Guidance



## Files

tools/bigcherry/tuning/recovery.py (RetuneRecommendation schema already landed here), future: a registry-diff module, a query surface over HTR02's witness store, an E2E-loss estimator

## Validation

No implementation validation yet -- this item's near-term acceptance criterion is simply staying correctly scoped as deferred until its stated prerequisite (real HTR01 usage history) exists.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

This item stays 'pending, not yet actionable' until HTR01 has real production/campaign usage data to design from -- do not pick this up speculatively. When it is picked up, re-open the negotiation with GPT rather than assuming this note captures a complete design; it explicitly does not (three of its four reasons are unimplemented by design).

## Change Log

- 2026-08-29T13:47:23.898965+00:00 (created-by): Created by agent
