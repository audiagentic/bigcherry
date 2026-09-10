---
id: RRBC01
order: 4
plan: run-reusable-build-campaign
state: pending
created-at: '2026-09-09T10:58:54.395708+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: P2
---

# Vulkan-aware device-visibility wiring for build's runtime smoke path

## Description

The frozen reusable-build-campaign item remains active with implementation/acceptance work outstanding and no terminal disposition.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Capability owner: run

DEPENDENCY CORRECTION (deeper repo-validated dev-gpt review, 2026-09-10): this item's stub Validation field originally said 'Frozen dependencies: RE30', which is BACKWARDS -- RE30 itself depends on RE31 (this item's own predecessor), not the other way around. RE30's active successor is TRBC01 (tuning-reusable-build-campaign plan), which is a downstream CONSUMER of this item's work (Vulkan-as-tunable-build-line evaluation), not a prerequisite of it. RRBC01 has no real upstream dependency among the run-*/build-*/tuning-* successor items and should be treated as foundational (matches its #4/#5 position early in the execution order).

## Code Samples & Guidance



## Files

tools/bigcherry/campaign/lane.py (smoke_environment_for_backend, implemented); tools/bigcherry/cli/build.py, re14_real_run.py, re15_acceptance_run.py (migrated callers); tools/tests/campaign/test_campaign_lane.py (7 tests). NOT yet done: per-lane backend derivation in campaign/planner.py's plan() -- cmd_build_new still computes one request-global environment from a single CLI flag.

## Validation

DONE 2026-09-10: 7 table tests (2 pre-existing HIP + 5 new: Vulkan, None, empty-error, unknown-backend-error, PATH-preserved) passing; full 19-test campaign-lane suite green. Device-safety mapping (Vulkan<->HIP enumeration order) NOT re-derived on real hardware -- still required before this item can be marked complete, per its own notes.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RE31
Migration: capability-rebaseline-v3-2026-09
Successor key: run-reusable-build-campaign-re31

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): GO, design sufficient as-is. CampaignLane.backend is the authoritative source of which visibility variable to emit; caller supplies a backend-neutral visibility selection and the adapter emits exactly one backend visibility env var. Reject conflicting inherited HIP/Vulkan visibility state rather than guessing which one wins. Re-derive current Vulkan<->HIP device-enumeration-order mapping fresh on real hardware before acceptance (do not trust the stale RE30-era mapping). Execution order: ranked #5 (removes the first concrete HIP-only assumption from build smoke; prerequisite for RRBC02/RRVP01 lane work).

CONFIRMED via deeper repo-validated dev-gpt review (2026-09-10): real gap verified against current code -- planner/lane has smoke_environment but no backend-neutral device-visibility adapter yet. One additional constraint to preserve: RE15's existing selector rule that HIP smoke deliberately sets HIP_VISIBLE_DEVICES only, not simultaneous HIP+ROCR selectors -- do not silently start setting both. Execution order shifts to #4 in the revised sequence.

IMPLEMENTED 2026-09-10: generalized campaign/lane.py's smoke_environment_for_hip_devices into smoke_environment_for_backend(backend, visible_devices) per RE31's own code sketch -- backend derives which env var is emitted (HIP_VISIBLE_DEVICES vs GGML_VK_VISIBLE_DEVICES via a small VISIBLE_ENV-equivalent mapping), rejects unknown backends and empty device strings, preserves RE15's real-hardware finding that only ONE visibility variable is ever set (never both HIP and ROCR). Migrated every real caller (cli/build.py's cmd_build_new -- currently still hardcodes backend="hip" since that's the only CLI-exposed flag today, re14_real_run.py, re15_acceptance_run.py, plus a stale comment in campaign/workers.py) to the new name -- no legacy alias kept, per project doctrine. 5 new table tests added alongside the 2 existing HIP tests (Vulkan translation, None->no key, empty->ValueError, unknown backend->ValueError, PATH preserved); all 19 campaign-lane tests pass.

NOT yet done (explicitly deferred, matches RE31's own critique of the current code that this item was filed to fix): cmd_build_new still computes ONE request-global smoke_environment from a single --hip-visible-devices CLI flag, rather than deriving backend PER PLANNED LANE in campaign_planner.py's plan() so a mixed HIP+Vulkan multi-lane build request gets correct per-lane visibility automatically. That planner-side threading is real remaining scope for this item (or naturally falls to RRVP01, which is already threading per-lane backend/stack identity through the same planner path) -- flag for whoever picks this back up rather than silently declaring it done. Device-safety mapping (Vulkan<->HIP enumeration order) has NOT been re-derived fresh on real hardware in this session -- still required before acceptance, per this item's own notes above.

## Change Log

- 2026-09-09T10:58:54.395708+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:03.776505+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.471692+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:36.484700+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:05.332242+00:00 (updated-by): Updated: order=5, priority='P2'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.943223+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.320402+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:18:53.401546+00:00 (updated-by): Updated: order=4, section:notes
- chg_20260910_002307_made-builds-device-visibility_2119
- 2026-09-10T00:23:07.296800+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:23:14.440795+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:27:29.287862+00:00 (updated-by): Updated: section:detailed_solution, section:files, section:validation
