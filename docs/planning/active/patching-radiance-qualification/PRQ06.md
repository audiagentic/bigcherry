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

# Radiance patch set: qualify or retire the 7 untested patches and delete the 4 superseded ones

## Description

Owner direction 2026-11-20: Radiance only, get current work and plans closed and patches validated; no legacy kept. engines/radiance/patches has 30 validated, 7 untested (rad0003_mtp_full_head_switch, rad0007_hc_pf2_fp8_down, rad0039_moe_heat_observer, rad0042_kv_b16_selftest, rad0043_expert_stats_protected, rad0044_expert_route_record, rad0045_expert_route_replay), 4 superseded (rad0001_drain_mover_before_arena_free, rad0006_heat_freeze_diagnostic, rad0034_prefill_concurrency, rad0035_prefill_order; no patch or recipe depends on them) and 2 diagnostic (rad0004_moe_remote_diagnostics, rad0010_routing_evidence).

## Steps

1. Delete the 4 superseded patch packages (bigcherry-patch-lifecycle: confirm the superseding patch is validated and in the production set, keep the supersession facts in the successor's README/evidence, remove the package and any recipe/experiment/test references); no archive copies in the tree.
2. For each untested patch decide with evidence: qualify to validated (bigcherry-patch-qualification: mechanics, apply/idempotence, the patch's own tests, the device/KLD/perf gate its kind needs) or retire (delete the package, migrate any caller). rad0042-0045 are VTP44 lab tools: qualify as diagnostic tools if they are kept (route record/replay, expert stats, KV selftest), or delete. rad0003/rad0007: find their purpose and current relevance on v1.17.0 first.
3. Confirm the 2 diagnostic patches are experiment-only (never in the production set) and still apply; else retire.
4. Re-run pin-bump/patch gates (patch-lint, patch-verify, patch-status) on main.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

patch-status shows no untested or superseded Radiance patches; patch-lint and apply/idempotence pass; each qualified patch has its gate evidence linked; each deleted patch is gone with no references (grep).

## Effort & Risk



## Standards



## Acceptance Criteria

Every Radiance patch is validated or diagnostic (experiment-only), none untested or superseded; no dead patch packages remain in the tree.

## Notes



## Change Log

- 2026-11-19T21:40:02.721878+00:00 (created-by): Created by agent
- 2026-11-19T21:40:21.043079+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
