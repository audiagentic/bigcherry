---
id: HTR03
order: 0
plan: hip-tune-recovery
state: pending
created-at: '2026-08-29T13:44:01.556347+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
---

# Versioned, configurable behavioral corpus + explicit workload-class applicability

## Description

HI143's regression-detection corpus is currently exactly ONE hardcoded, frozen prompt (tools/bigcherry/tuning/fixtures/hi141_qwen38_27b_mtp_4096_v1.txt), selected via a hardcoded _default_behavioral_corpus() function that always returns [load_hi141_regression_vector()] -- adding a newly-discovered regression scenario requires a code change. Applicability is a single boolean require_mtp inferred by string-matching '--spec-type' in the runtime profile's server_args -- not a general concept of 'decision-sensitive workload class'.

Adversarially designed with GPT (session ses_330ae3c055084f38, 2026-08-29) to fix this WITHOUT overbuilding -- GPT explicitly rejected building any trigger-plugin API, live/production-traffic monitoring, or automatic corpus rotation right now as premature (no evidence yet of real requirements for those), while confirming the corpus/applicability configurability itself is NOT premature given the current hardcoding is already an acknowledged, real gap (not speculative future-proofing).

## Steps

1. Define three independent identities, per GPT's explicit correction to conflating them: `behavioral_gate_contract_version` (comparison SEMANTICS -- hard_fail/exact_pass/behavior_changed three-state contract itself, mirrors HI67's CONTRACT_VERSION pattern), `corpus_schema_version` (the manifest FILE FORMAT), and `corpus_edition_id` + `content_digest` (the exact curated CONTENTS -- which vectors, with what parameters). Do not let contract_version also mean corpus contents (GPT explicit warning).
2. Build a YAML (or JSON, matching this repo's existing config style) corpus manifest format, e.g.:
   schema_version: 1
   edition: qwen38-production-v1
   vectors:
     - id: hi141-mtp-4096-v1
       prompt_file: hi141_qwen38_27b_mtp_4096_v1.txt
       prompt_sha256: <digest>
       n_predict: 128
       seed: 42
       applies_to: [mtp-speculative]
       scenario: long-prefill-mtp
       provenance: HI141
   An edition is immutable once published: any add/remove/content/parameter change requires a NEW edition id + content digest, never an in-place mutation -- this is what keeps historical campaign results reproducible (a receipt recording 'edition qwen38-production-v1' must always resolve to the exact same vectors it did originally).
3. Add explicit `RuntimeProfile.behavioral_classes: list[str]` metadata (e.g. ['mtp-speculative']) to config/recipes.toml's runtime-profile definitions -- stop inferring workload semantics from string-matching CLI args (GPT explicit: 'RuntimeProfile currently only stores argv/context/VRAM metadata; semantic workload class should become explicit configuration').
4. Applicability resolution: a vector applies to a campaign iff its `applies_to` list intersects the runtime profile's `behavioral_classes`. Use a DATA-DEFINED closed vocabulary (a declared class registry the config validates against), NOT a Python enum (every new workload class would require a code deployment) and NOT unconstrained free-form tags (a typo like 'mtp-speculativ' would silently drop required coverage with no error). Validate both runtime profiles' and vectors' class tags against the registry at config-load time.
5. Fail closed on: an unknown class tag anywhere (config/manifest error, never a silent skip), a campaign whose runtime profile declares a behavioral_classes value for which the resolved corpus edition contains zero applicable vectors (this must be loud, not quietly treated as 'nothing to check'), or an unresolvable/missing corpus edition reference.
6. Persist the resolved corpus edition id + content digest + the exact selected vector ids/digests actually used into the campaign receipt (WorkflowReceipt) and the behavioral-gate.json report -- so a past campaign's exact detection coverage is always independently reconstructable, not just assumed from whatever the current code happens to load.
7. Architectural seam only (GPT explicit: do not build the consumers yet): split today's _stage_replay_validate so it no longer owns corpus construction directly -- extract a CorpusLoader (resolves edition + applicable vectors from config) and keep a BehavioralGateEvaluator (evaluates native vs candidate cache against whatever vectors it's given) as the reusable unit a hypothetical future periodic/live-validation caller could invoke without touching corpus or gate internals. Do NOT build a TriggerStrategy interface, scheduler, production-traffic sampler, or event bus -- GPT was explicit these have no demonstrated requirement yet.

## Detailed Solution & Technical Design

GPT's explicit premature-abstraction boundary (adopt verbatim as this item's scope fence):

NOT premature (build now): external corpus manifest; immutable editions/digests; explicit runtime behavioral classes; corpus applicability resolver; receipt provenance.

PREMATURE (do not build): arbitrary policy-expression language; hierarchical/tag inheritance; dynamic predicates against CLI flags/model metadata; trigger plugin framework; production-traffic ingestion/sampling; automatic corpus rotation.

GPT's recommended v1 boundary, adopted as this item's acceptance criteria: versioned manifest + immutable vector files/digests + data-defined behavioral-class vocabulary + explicit RuntimeProfile.behavioral_classes + deterministic applicability resolver + corpus identity persisted in gate report/receipt + existing promotion-time evaluator only (no new trigger points). Everything under 'defer' above is explicitly OUT of this item's scope, tracked only as a named future possibility, not a design obligation here.

## Code Samples & Guidance



## Files

tools/bigcherry/tuning/behavioral_gate.py, tools/bigcherry/tuning/workflow.py, tools/bigcherry/tuning/fixtures/ (new manifest file), config/recipes.toml (behavioral_classes field), tools/bigcherry/core/config.py (RuntimeProfile schema)

## Validation

Offline tests: manifest parsing/validation (schema errors, unknown class tags fail closed), applicability resolution (class intersection logic, zero-applicable-vectors-for-declared-class fails loud), edition immutability (content digest changes iff vectors/params change). Real-hardware regression test: confirm the existing HI141 vector, now loaded via the new manifest path instead of the old hardcoded function, still produces the exact same hard_fail result against the same known-guilty candidate (no behavior change from the refactor itself).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Directly informed by walking through, in conversation, exactly how HI143's current trigger fires and its real boundaries (single hardcoded vector, MTP-boolean-only, promotion-time-only) -- this item closes the corpus/applicability half of that gap. The 'promotion-time-only, no live monitoring' boundary is DELIBERATELY not addressed here per GPT's premature-abstraction verdict; if a real future need for periodic/live checking emerges, it should reuse the CorpusLoader/BehavioralGateEvaluator seam this item creates, as its own separate, later-scoped item -- not be designed speculatively now.

PROVENANCE HARDENINGS ADDED (2026-08-30), per GPT review incorporating real HTR01 usage history (three real-hardware runs, two real bugs found only by later deep-diving actual committed logs -- these additions exist specifically so a FUTURE deep-dive can reconstruct exactly what was checked without needing to re-derive it from whatever code exists at review time):

A. Bind validation to the exact cache artifact: record validated_cache_digest, behavioral_gate_report_path, behavioral_gate_report_digest in the receipt/report -- a path alone (today's WorkflowReceipt.dispatch_cache_path) is insufficient since the same path can be overwritten by a later run.

B. Bind to exact runtime-profile CONTENTS, not just its name: record runtime_profile_name AND runtime_profile_digest (canonical hash over server_args, tune_context, production_context, min_free_vram_bytes_per_device, behavioral_classes) plus behavioral_classes itself. Today's receipt only records runtime_profile_name -- the named profile's contents can change later under the same name with no detectable trace.

C. behavioral-gate.json becomes a full resolved execution snapshot, not just a name/verdict summary. Receipt stays compact (behavioral_gate_contract_version, corpus_schema_version, corpus_edition_id, corpus_content_digest, selected_vectors:[{id,digest}], validated_cache_digest, runtime_profile_digest, behavioral_gate_report_digest); behavioral-gate.json itself stores PER VECTOR: id, vector_digest, prompt_sha256, n_predict, seed, applies_to, scenario, provenance, verdict, native_token_digest, candidate_token_digest, token_count, native_draft, candidate_draft, first_output_divergence. Today's report only stores name/verdict/draft totals/divergence -- insufficient for a future reviewer to reconstruct exact vector parameters without re-deriving them from the manifest/parser as it exists AT REVIEW TIME (which may have changed).

D. Applicability (behavioral_classes) and REQUIRED TELEMETRY are separate concepts -- do not conflate them. Add an explicit, separate, small closed vocabulary `requirements: [mtp-telemetry]` per vector in the manifest, replacing the current implicit assumption (run_vector(require_mtp=True) as if every vector needs MTP telemetry) which does not generalize once HTR03 allows non-MTP behavioral classes. A new behavioral class reusing EXISTING telemetry stays pure data (no code change); a genuinely new telemetry requirement needs new evaluator code, which is correct and expected.

Explicitly reconfirmed as still out of scope for this item (GPT, unchanged from original design): raw-log archival, trigger framework, live/production monitoring, giant receipt snapshots -- structured digests + the resolved gate report are sufficient.

SEE ALSO: HTR06 (new, 2026-08-30) -- a related but explicitly SEPARATE investigation into whether tune-time synthetic candidate RANKING (not correctness) predicts real E2E performance ranking, raised by the user and adversarially scoped with GPT as a bounded empirical study, not a mandate to change selection logic. HTR03's corpus vectors may later be reused as HTR06's real workload shapes, but the two investigations' results must never contaminate each other.

## Change Log

- 2026-08-29T13:44:01.556347+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260829_134919_the-tuning-system-can-now-reco_8101
- 2026-08-29T13:49:19.490008+00:00 (updated-by): Updated: section:ledger-events
- 2026-08-29T23:58:18.136268+00:00 (updated-by): Updated: section:notes
