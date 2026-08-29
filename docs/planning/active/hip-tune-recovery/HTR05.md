---
id: HTR05
order: 0
plan: hip-tune-recovery
state: pending
created-at: '2026-08-29T13:56:39.699257+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
---

# Empirical multiplicity-correction investigation (BH vs Holm vs BY) -- study first, no production change

## Description

Adversarially reviewed with GPT (session ses_330ae3c055084f38, 2026-08-29). Question raised: tune_promotion.py hardcodes Benjamini-Hochberg (BH) as its multiple-hypothesis correction across every non-native provisional winner in a campaign -- is this the right default, and should it be pluggable?

GPT's verdict, adopted verbatim as this item's governing conclusion: HI141 is NOT evidence against BH. HI141's guilty candidate had a real ~23% measured latency win -- its promotion was not a statistical false discovery, it was a genuine speed improvement with a real-data behavioral defect BH has nothing to do with. 'BH controls false claims of PERFORMANCE improvement. HI143 controls BEHAVIORAL safety.' These are orthogonal concerns; a stronger multiplicity correction would very likely still have promoted the exact same candidate. Do NOT change BH now on the basis of HI141.

## Steps

1. Do NOT change tune_promotion.py's BH correction as a result of this item alone -- that requires the empirical study below to actually demonstrate a real problem first, per GPT's explicit 'no demonstrated evidence currently justifies changing production BH' verdict.
2. Real open, testable question GPT identified: BH assumes independence or positive-regression-dependence (PRDS) among the null hypotheses' p-values. BigCherry's own candidates share structural similarity (many mmvq:* variants differing only in geometry parameters) AND share real measurement-time confounds (GPU thermal/clock state, measurement sequencing, a common native baseline) that COULD induce dependence violating BH's assumptions -- but GPT was explicit this is unproven, not established, by the mere existence of structural similarity. tune_promotion.py's existing simulate_null_fdr() only validates against independent uniform p-values, so it specifically does NOT test BigCherry's real dependence structure -- a real, concrete gap.
3. Design and run an offline empirical investigation: using archived real campaign measurement data (native-twin/known-null comparisons, which preserve actual cross-signature measurement correlation, unlike synthetic uniform p-value simulation), compare empirical false-discovery behavior under BH, Benjamini-Yekutieli (BY, valid under arbitrary dependence, more conservative), and Holm-Bonferroni (family-wise error control, philosophically aligned with 'one false speed promotion is worse than several missed ones' if BigCherry ever explicitly adopts that policy stance -- GPT explicit this must be a deliberate policy decision, not a reaction to HI141's unrelated semantic failure).
4. ONLY if that study demonstrates a material real risk under BH given BigCherry's actual dependence structure: introduce a versioned MultiplicityPolicy interface (name/version + adjust(hypotheses) -> adjusted values, persisted in promotion_policy identity the same way ranking policy identity already is) with bh-v1/holm-v1/by-v1 as candidates.
5. Explicit, permanent constraint (GPT emphatic, do not revisit without a very strong justification): the correction METHOD may eventually become pluggable (step 4); the FAMILY DEFINITION (what set of hypotheses gets corrected together) must NOT become an arbitrary configurable knob (e.g. 'per-op', 'per-shape', 'per-workload'). Finer partitioning always mechanically weakens the overall statistical guarantee and creates an incentive to choose whatever partition yields the most promotions -- family boundaries must stay fixed/prospective, defined by the statistical contract itself, never a runtime choice.

## Detailed Solution & Technical Design

Direct implication for HTR04 (retune escalation), explicit GPT correction, recorded here AND cross-referenced into HTR04's own notes: a future targeted single-signature retune CANNOT become statistically valid merely by declaring 'family = this one signature' after the fact -- that reintroduces the original experiment's already-spent multiplicity budget as a new, uncontrolled sequential-testing problem. GPT's listed valid future approaches (any one, not yet designed): recompute the ENTIRE original family with fresh, comparable evidence; a predefined hierarchical/group error-allocation scheme; alpha-spending/online-FDR semantics; or explicitly defining a targeted retune as a wholly NEW experiment whose results do not inherit any claim from the old family without an explicit, specified composition rule. HTR04 must solve this separately and explicitly -- it is not a side effect of HTR05's own scope.

## Code Samples & Guidance



## Files

tools/bigcherry/tuning/tune_promotion.py (benjamini_hochberg(), simulate_null_fdr() -- read/study, not modified by this item alone)

## Validation

This item's own deliverable IS the empirical study and its writeup (a real comparison of BH/BY/Holm false-discovery behavior against real archived campaign measurement correlation structure) -- not code. A follow-up implementation item should only be created if that study's own findings warrant it.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

This item exists to prevent two failure modes symmetrically: (a) reactively weakening/replacing BH because of HI141 despite HI141 having nothing to do with statistical multiplicity (the mistake this item's own negotiation avoided), and (b) HTR04 quietly assuming a narrow-family shortcut is fine for targeted retunes without ever confronting the real statistical problem GPT identified. Pending: no work has started; this item is 'investigate first' by design, not 'implement.'

## Change Log

- 2026-08-29T13:56:39.699257+00:00 (created-by): Created by agent
