---
id: PRQ06
order: 0
plan: patching-radiance-qualification
state: pending
created-at: '2026-11-19T21:40:02.721878+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Radiance patch set: qualify or retire the 7 untested patches and archive the 4 superseded ones

## Description

Owner direction 2026-11-20: Radiance only, get current work and plans closed and patches validated; no legacy kept. engines/radiance/patches has 30 validated, 7 untested (rad0003_mtp_full_head_switch, rad0007_hc_pf2_fp8_down, rad0039_moe_heat_observer, rad0042_kv_b16_selftest, rad0043_expert_stats_protected, rad0044_expert_route_record, rad0045_expert_route_replay), 4 superseded (rad0001_drain_mover_before_arena_free, rad0006_heat_freeze_diagnostic, rad0034_prefill_concurrency, rad0035_prefill_order; no patch or recipe depends on them) and 2 diagnostic (rad0004_moe_remote_diagnostics, rad0010_routing_evidence).

## Steps

1. Implement the retired-patch archive (docs/reference/patches/PATCH_SYSTEM.md 'Retired patches: archive, never delete'): engines/radiance/patches/archive/ with ARCHIVE.md, and make the patch registry, patch-lint, pin-bump, recipes and builds skip archive/ (tests first).
2. Archive the 4 superseded/replaced patches (rad0001_drain_mover_before_arena_free, rad0006_heat_freeze_diagnostic, rad0034_prefill_concurrency, rad0035_prefill_order) unchanged, one ARCHIVE.md row each with reason, successor and evidence links; transition record + ledger event.
3. For each untested patch decide with evidence: qualify to validated (bigcherry-patch-qualification) or retire (set rejected/abandoned with the reason, then archive). rad0042-0045 are VTP44 lab tools: qualify them as diagnostic tools if they are kept. rad0003/rad0007: establish purpose and relevance on v1.17.0 first.
4. Confirm the 2 diagnostic patches (rad0004, rad0010) are experiment-only and still apply.
5. Re-run patch-lint, patch-verify and patch-status on main.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

patch-status lists no untested or superseded patch in the active Radiance set; archive/ holds the retired packages unchanged with ARCHIVE.md rows; registry/lint/pin-bump tests prove archive/ is never composed; patch-lint and apply/idempotence pass.

## Effort & Risk



## Standards



## Acceptance Criteria

Every active Radiance patch is validated or diagnostic (experiment-only); every retired patch is kept unchanged under engines/radiance/patches/archive/ with its reason and evidence indexed; none was deleted.

## Notes



## Change Log

- 2026-11-19T21:40:02.721878+00:00 (created-by): Created by agent
- 2026-11-19T21:40:21.043079+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
- 2026-11-19T21:47:53.617559+00:00 (updated-by): Updated: section:title, section:steps, section:validation, section:acceptance_criteria
