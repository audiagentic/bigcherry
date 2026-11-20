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

## Description

Audit the standard producer build set before attempting to remove tune/replay builds or campaign identity fields. The original claim that tune/replay binaries are never executed is contradicted by current activation and e2e campaign paths. Preserve every live tune/replay consumer and historical evidence. Any reduction must be based on a producer-path call graph proving a specific role is unused, with identity/schema/verifier migration in the same change.

## Steps

1. Map every standard producer and dispatcher call path that consumes tune_bin, replay_bin, tune_server, replay_server, tune_bench and replay_bench, including activation probes, e2e campaign operations and replay validation.
2. Classify each role by whether it is built, identified, launched or required by a non-standard command; do not infer non-use from the ordinary benchmark lane alone.
3. Retain all roles with a live consumer. Only propose removing a build when the exact producer/dispatcher call graph proves no activation, correctness, campaign, or replay path uses it.
4. If a role can be removed, migrate campaign identity digests and evidence schema atomically, update new-record verification, preserve historical-record verification, and retain tune/replay builds on the tuning campaign path that executes them.
5. Measure build-count and behavior changes in offline fixtures and a representative producer session before accepting any saving.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Static call-graph evidence must cover campaign/scaffold/producer activation, e2e campaign and replay entry points. Tests must prove required tune/replay consumers still launch and identity records remain complete; historical records remain verifiable after any schema change. Any proposed reduced build set must show exact producer behavior and unchanged verdict/evidence semantics. No tests or hardware runs were performed during this planning correction.

## Effort & Risk



## Standards



## Acceptance Criteria

No live tune/replay execution or evidence-identity consumer is removed. The producer's required build set is proven from all activation, correctness, campaign and replay paths. A role is dropped only with an exact call-graph proof, atomic identity/schema/verifier migration, retained historical verification, and tests showing required flows are unchanged. The original build-count saving is not accepted as a premise.

## Notes

Filed 2026-09-25 alongside the ccache fix (9c4dee01): ccache was at the 5 GiB default with 3658 cleanups and a 55% hit rate because content-addressed worktree paths defeated cross-tree hits.

2026-09-27: Confirmed step 1 -- grepped every caller of _build_standard_campaign_scaffold (only two: validation_campaign.py's legacy run() path, and producer.py's generic producer path). producer.py never references tune_bin/replay_bin/tune_build_evidence/replay_build_evidence anywhere -- confirmed dead weight on that path. run()'s _prepare_standard_campaign genuinely executes tune_server/replay_server via e2e_smoke_campaign.Campaign and uses tune_bin as the activation-probe binary, so it must keep all 5 builds.

Blocked on a deeper finding than scoped: campaign_build_identities and the evidence record writer are ONE shared function (evidence.py, hardcoded "record_schema_version": 4, CAMPAIGN_BUILD_ROLES=("tune","replay","stock") required exactly) used by BOTH paths -- there is no per-path writer to special-case. Doing this correctly needs a real schema version bump (v5) with path-gated role-set validation (producer path: stock+base only; run() path: unchanged tune/replay/stock), threaded through _build_standard_campaign_scaffold (new build_tune_replay: bool param), producer.py's identity-digest call sites, and evidence.py's writer + verify-evidence reader (record_version in (3,4) branch at evidence.py:1254 would need a new branch for 5, keeping 3/4 readable as history per doctrine).

Deliberately NOT implementing this now: it's a genuine schema fork touching the exact evidence-writing code path the live perf-lane hardware campaigns (t-1268b-*, t-1265c-*) are using this session, and every other schema-version fork in this codebase (PA28-31, VA07, VA18) shows a GPT design review attached before landing. Pick this up in a dedicated window with no live hardware campaign in flight, and get a design review on the v5 role-set split before implementing.

## Change Log

- 2026-09-25T12:43:53.159999+00:00 (created-by): Created by agent
- 2026-09-27T12:36:15.733308+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4254
- 2026-11-20T04:13:01.931953+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
