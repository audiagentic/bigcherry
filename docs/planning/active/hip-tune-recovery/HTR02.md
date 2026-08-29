---
id: HTR02
order: 0
plan: hip-tune-recovery
state: pending
created-at: '2026-08-29T13:28:32.962287+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
---

# Durable BehavioralFailureWitness records (not a candidate blacklist)

## Description

Adversarially negotiated with GPT (session ses_330ae3c055084f38, 2026-08-29) in response to the user's question: once HTR01's recovery search confirms a candidate+signature causes a real behavioral regression, should that be persisted so future campaigns skip re-discovering the same failure, or is that too risky given builds/drivers/hardware can change?

GPT's answer: persist it, but explicitly NOT as a universal 'candidate+signature = bad' blacklist -- HI141 itself disproves that abstraction (the guilty candidate alone fails; a different real ensemble containing the SAME candidate+signature passed). Name and design it as a BehavioralFailureWitness: a scoped record of one real observed failure under one full, specific context -- never an intrinsic property of the candidate.

## Steps

1. Define the BehavioralFailureWitness schema (see code_samples) with the full context-identity fields GPT specified -- candidate_digest, signature_digest, source_slice_id (exact effective source tree including applied patches, not just branch/revision), build_plan_id, model/content identity, runtime-profile + exact MTP/server args, hardware/topology identity, ROCm/runtime/driver identity, corpus/vector identity, the full tested assignment digest, the failing-group/minimal-set if HTR01 identified one, verdict, first-divergence data, evidence timestamp/schema version.
2. Implement the three-level reuse policy GPT specified, exactly (do not weaken to a simpler two-level or single-level scheme):
   - Exact full failure identity match (identical complete assignment + identical full context) -> MAY skip re-testing that exact failed assignment; feed its recorded verdict directly into HTR01's recovery search as a prior.
   - Same candidate/signature + exact environment, but a DIFFERENT candidate ensemble -> strong negative prior for HTR01's alternative-ordering (downrank this candidate's search priority) -- MUST NOT hard-exclude it from consideration.
   - Different source/build/model/ROCm/driver/hardware/runtime identity in any dimension -> historical diagnostic evidence only, zero automatic exclusion or downranking effect.
3. Validity/expiry: use content/context identity invalidation, NOT time-based expiry, as the primary mechanism (GPT explicit: age may be recorded as metadata, but a record becomes inapplicable automatically the moment any relevant identity dimension changes, not after some elapsed duration). 
4. Explicitly do NOT let a ddmin-derived failing GROUP be interpreted as 'every future superset containing this group also fails' -- record and use it only as evidence about the exact tested assignment(s), consistent with HI141's proven non-monotonic masking behavior.
5. Hard invariant, enforced in code, not just documentation: NO historical BehavioralFailureWitness record, at any confidence level, may ever cause a campaign to SKIP HI143's final full-cache behavioral gate on the assignment it is about to publish. Witness records may only ever (a) skip re-testing an assignment identical in every context dimension to one already tested, or (b) reorder/deprioritize HTR01's search -- never bypass the oracle.

## Detailed Solution & Technical Design

Naming (GPT explicit, adopt verbatim): BehavioralFailureWitness, never KnownBadCandidate -- the name itself should make misuse (treating it as an intrinsic candidate property) harder to fall into by accident.

Dependency gap GPT flagged as a real, currently-missing prerequisite: 'BigCherry currently lacks a sufficiently complete GPU/software behavioral fingerprint in core/gpu.py -- it only queries VRAM for preflight. Do not pretend device indices constitute hardware identity.' This item's hardware/topology/ROCm/driver identity fields depend on that fingerprint existing -- check tools/bigcherry/core/gpu.py's current state and scope a fingerprinting improvement here (or as a split-out prerequisite item) rather than approximating identity with whatever's cheapest (e.g. device index or GPU model string alone), since an incomplete fingerprint silently degrades the whole scheme's safety by making distinct real environments look identical.

## Code Samples & Guidance



## Files

tools/bigcherry/tuning/ (new module, e.g. failure_witness.py), tools/bigcherry/core/gpu.py (hardware/driver identity gap), sql/dispatch-db.sql (new table if persisted in the existing sqlite store, matching correctness_evidence's own pattern)

## Validation

Offline tests for the three-level reuse policy's classification logic (exact-match / same-candidate-different-ensemble / different-context, using synthetic context tuples) and for the hard invariant that no code path can construct a published cache without an HI143 full-gate PASS record attached, regardless of witness state.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Depends conceptually on HTR01 (this is where HTR01's recovery Observations get persisted and consulted from) but is a genuinely separate concern per GPT's explicit instruction not to conflate them -- HTR01 can and should be built and validated first using only in-memory/per-campaign state, with this item's durable cross-campaign persistence layered on after.

## Change Log

- 2026-08-29T13:28:32.962287+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260829_132841_planned-the-next-improvement-t_2822
- 2026-08-29T13:28:41.383575+00:00 (updated-by): Updated: section:ledger-events
