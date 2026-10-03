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

Every standard producer session builds five trees (tune, replay, stock, control, validation-subject). On the producer path tune/replay binaries are never run; they only feed campaign_build_identities into the campaign identity digest and the evidence record. Dropping them saves ~40% of per-session build work.

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
