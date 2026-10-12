---
id: PVPS04
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-25T12:43:53.159999+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# Skip tune/replay builds on producer campaigns (they are built but never executed)

## 2026-10-12 authoritative implementation audit (supersedes old 40%-time claim)

**Disposition: keep PVPS04 pending, default existing scaffold until a versioned, fail-closed evidence migration is ready.** In `tools/bigcherry/patch/campaign/scaffold.py::_build_standard_campaign_scaffold` (~505-623), `_build_tune_and_replay_trees` (~270-357) is unconditional. `_build_parity_trees` (~360-503) builds stock, control, validation-subject and optionally a **distinct base** if `sources.base_src != control_src`. `campaign/producer.py::_prepare_standard_producer_campaign` (~1596-1710) consumes stock/control/subject/base and uses tune/replay **only as build identities**, never as executables. By contrast `validation_campaign.py::_prepare_standard_campaign` (~505-622) actually executes tune/replay through `e2e_smoke_campaign.Campaign` and uses tune for activation probes; this legacy path must retain them.

**Theoretical saving:** eliminate **2 of 5 or 2 of 6 build-tree invocations** (40% or 33.3% by count), depending on whether PVPS03's base arm aliases control. This is NOT measured build wall time, compilation work, power or inference throughput: CMake and ccache can reuse work across invocations. No actual build or benchmark was run.

### Exact change set and ownership/lifetime

1. `campaign/scaffold.py::StandardCampaignScaffold` and `_build_standard_campaign_scaffold`: add a private explicit `mode: Literal["producer","tune-replay"]` chosen by the existing caller (not a CLI flag). For producer mode, skip `generate_registry` and `_build_tune_and_replay_trees`; represent tune/replay binaries/evidence as absent, never aliased to stock or fabricated. For legacy tune-replay mode, keep all builds, generated registry and activation/bench paths unchanged. Preserve the stock, control, subject, optional base, source/composition/idempotence, `assert_validation_subject_parity`, and `reference_ladder_bins` behaviour. Accessing absent tune/replay in legacy mode must fail before execution.
2. `campaign/producer.py::_prepare_standard_producer_campaign` passes producer mode; `validation_campaign.py::_prepare_standard_campaign` passes tune-replay mode. Self-contained producers with `scaffold=None` are unaffected. `campaign_build_identities` becomes **exactly** `{"stock"}` for producer, and `{"tune","replay","stock"}` for legacy. The separate `validation_build_identities={"control","subject"}` remains mandatory; optional base belongs to PVPS03's ladder, not a fake campaign role.
3. `patch/evidence.py::make_record` (~800-980) writes **record_schema_version=5** with required `campaign_mode`. Enforce exact mode/role sets at write and read: v5 producer requires stock only; v5 tune-replay requires all three. Unknown/missing mode, extra/missing roles, malformed/foreign build identities, or v5 legacy top-level `build_identities` => INVALID. Do not change activation, correctness, multi-contract, hardware, artifact hash, or promotion gates. Update `_record_qualifies` (~1148-1345), `_record_qualifies_for_benched` (~1350-1500), and every other schema-version gate. Preserve v1-v4 **read-only historical** semantics, especially v4's exact three-role and plural-contract requirements. No silent legacy fallback or record rewrite.
4. `model_free_campaign_identity_digest` and `producer_campaign_identity_digest` (~190-335) currently reject any campaign role set except tune/replay/stock. Introduce separate **v2 producer identity schema labels** including explicit producer mode and the actual stock build, immutable patch/source/arch/base revision, and model/corpus/producer-input bytes where applicable. Keep v1 digests bit-identical for historical records. Legacy `CampaignIdentityContext` retains its real three-build binding. Ensure deterministic `record_digest`, `write_record`, `patch-verify-evidence`, and stale/withdrawal checks. Do not substitute cached historical tune/replay identities.
5. PVPS03 owns the four-arm reference ladder and its build-cache/identity semantics; PVPS01 owns multi-arm policy; PEC02 owns evidence completeness. PVPS04 owns only producer build-mode elimination and the required versioned provenance. No new runner, allocator, scheduler, build registry, telemetry system or benchmark surface.

### Cheap discriminator and implementation/acceptance gates

- Add `tools/tests/patch/test_pvps04_producer_build_roles.py`: mocked build call counts for producer (stock/control/subject + conditional base, **no** tune/replay/registry) and legacy (tune/replay retained), with base=control and base!=control, multiple sequential producers, failures before record persistence, and legacy trace-probe execution.
- Extend `test_patch_evidence_schema_v3.py`, `test_patch_evidence_va18.py`, `test_validation_producer_execution.py`, `test_reference_ladder.py` and `test_patch_validation_campaign_va18.py`: producer/legacy exact role sets, missing/extra/foreign roles, model-free and model/corpus-bound digest changes, malformed identities, v4 history unchanged, multi-contract and ported-benched gates, identical stock/control/subject parity, and complete artifact binding. No GPU is required for these discriminators.
- Run offline pytest and `patch-verify-evidence` on a committed historical v4 record plus a synthetic v5 record. Only then run one **uncontended** real producer session and matched cold/warm compilation A/B (prefer four paired runs) with fixed source/arch/cache; record actual CMake invocations, compile counts, wall time and complete evidence. Require zero false promotion decisions, no loss of correctness/reference arms, and no >1% validation execution regression. If v5 cannot be atomic, a producer consumes tune/replay, or historical records stop verifying, **REJECT** and retain the existing scaffold.

**Upstream/fork decision:** llama.cpp b11552/current CMake does not define BigCherry's tune/replay provenance; do not import a kernel or runtime scheduler for this. vLLM `setup.py` has optional precompiled-extension reuse (build-cost precedent only); SGLang's runtime architecture is not a campaign identity mechanism. No transferable AMD inference speedup. Sources: https://github.com/ggml-org/llama.cpp/blob/b11552/ggml/src/ggml-hip/CMakeLists.txt ; https://github.com/vllm-project/vllm/blob/main/setup.py ; https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/engine.py .

**Actual audit validation:** 15/15 source-static assertions and 16/16 disposable Python host-model role/digest assertions passed. No repository pytest, HIP/C++ build, timed build, GPU execution or hardware benchmark ran. Historical September notes below remain provenance; their five-build/40%-time statements are superseded here.

## Description

Every standard producer session builds five trees (tune, replay, stock, control, validation-subject). On the producer path tune/replay binaries are never run; they only feed campaign_build_identities into the campaign identity digest and the evidence record. Dropping them may save two build variants, subject to the v5 evidence gate above; no measured wall-time saving is established.

## Steps

1. Confirm no producer or dispatcher path executes tune_bin/replay_bin.
2. Make campaign_build_identities producer-path = {stock} (plus the PVPS03 base arm), migrate the identity digest and evidence schema (no legacy fallback), update verify-evidence for new records while keeping historical records verifiable as history.
3. Keep tune/replay builds for the tuning campaign path that actually runs them.
4. Tests + one hardware session showing 3 builds (stock, control, subject).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline suite; patch-verify-evidence on existing records still reports them as historical; a new session persists evidence with the new identity shape.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Filed 2026-09-25 alongside the ccache fix (9c4dee01): ccache was at the 5 GiB default with 3658 cleanups and a 55% hit rate because content-addressed worktree paths defeated cross-tree hits.

2026-09-27: Confirmed step 1 -- grepped every caller of _build_standard_campaign_scaffold (only two: validation_campaign.py's legacy run() path, and producer.py's generic producer path). producer.py never references tune_bin/replay_bin/tune_build_evidence/replay_build_evidence anywhere -- confirmed dead weight on that path. run()'s _prepare_standard_campaign genuinely executes tune_server/replay_server via e2e_smoke_campaign.Campaign and uses tune_bin as the activation-probe binary, so it must keep all 5 builds.

Blocked on a deeper finding than scoped: campaign_build_identities and the evidence record writer are ONE shared function (evidence.py, hardcoded "record_schema_version": 4, CAMPAIGN_BUILD_ROLES=("tune","replay","stock") required exactly) used by BOTH paths -- there is no per-path writer to special-case. Doing this correctly needs a real schema version bump (v5) with path-gated role-set validation (producer path: stock+base only; run() path: unchanged tune/replay/stock), threaded through _build_standard_campaign_scaffold (new build_tune_replay: bool param), producer.py's identity-digest call sites, and evidence.py's writer + verify-evidence reader (record_version in (3,4) branch at evidence.py:1254 would need a new branch for 5, keeping 3/4 readable as history per doctrine).

Deliberately NOT implementing this now: it's a genuine schema fork touching the exact evidence-writing code path the live perf-lane hardware campaigns (t-1268b-*, t-1265c-*) are using this session, and every other schema-version fork in this codebase (PA28-31, VA07, VA18) shows a GPT design review attached before landing. Pick this up in a dedicated window with no live hardware campaign in flight, and get a design review on the v5 role-set split before implementing.

## Change Log

- 2026-09-25T12:43:53.159999+00:00 (created-by): Created by agent
- 2026-09-27T12:36:15.733308+00:00 (updated-by): Updated: section:notes
