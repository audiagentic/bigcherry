---
id: THA05
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:48:25.987562+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# bigcherry tune-campaign: single-command record->tune->correctness->promote->replay orchestrator

## Description

Deliver the production single-command tune-campaign orchestrator for record→tune→correctness→promote→replay with deterministic receipts and configurable runtime profiles.

## Steps

Implement workflow stages and thin CLI; reuse server lifecycle helpers; expose source_root; persist per-stage build/source/artifact identities and effective context; use named production profiles with bounded tune context and explicit preflight; replay from tune build manifest; reproduce the validated dual-XTX Qwen3.8-27B baseline and coverage behavior without legacy shared-checkout mutation.

## Detailed Solution & Technical Design

Production orchestrator, not smoke campaign reuse. Fail closed on unsafe VRAM/context and preserve campaign APIs. Receipt must include run/source/build IDs, manifest, cache, promoted count and profile.

## Code Samples & Guidance



## Files

tools/bigcherry/tuning/workflow.py; cli/tuning.py/main.py; CampaignLaneResult; receipt/build JSON; runtime profiles; release persistence idempotency.

## Validation

Single-command real dev record/tune/correctness/promote/replay, receipt completeness, 20/39-style promotion baseline where reproducible, bounded context and replay coverage.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

One deterministic command completes all stages with immutable receipt/provenance, no shared-checkout interference, explicit effective context, and no regression against validated dual-XTX baseline.

**Transferred THA06 acceptance (independent gate):** Retain explicit --retain-from receipt/provenance input in the existing tune-campaign orchestration only: perform tuning-free record-only on the new pinned build, then project prior winners via the maintained replay_projection and per-candidate capability/implementation equivalence, not global manifest equality or indiscriminate revision bypass. Include model/profile/architecture/build/context and previous decision identity. Re-attest each compatible winner under current candidate can_execute/quality/launch gates, quarantine incompatible or unverified histories, and retune *only* affected signatures. All new build artifacts pass existing source/planner/admission, no second global cache or cross-model store; persist per-candidate retained/retune/denied receipt and fail closed on missing provenance.

**Transferred THA27 acceptance (independent gate):** Within THA05's maintained candidate assessment/report pipeline, join recorded call counts, timed per-signature/family kernel measurements and measured rocprofv3 matmul fraction/critical-path share by stable kernel mapping. Compute call-weighted predicted E2E movement, uncertainty/CI bounds and number of order-balanced repetitions needed to resolve it; use settled no-flush B2 preconditioning and avoid treating isolated speedup as guaranteed end-to-end benefit. Run the required A/B count or explicitly record BELOW_RESOLUTION/NO_MEASURED_FRACTION when the effect is unresolvable. Every promoted tune win links a reproducible prediction, measured decode fraction, CI and actual E2E decision; share the source of this report with THA16 rather than implementing an alternate profiler.

**Transferred THA31 acceptance (independent gate):** Under THA05's screen stage, compute same-kernel canary_screen_pct *before* retention and derive band=clamp(2×canary_screen_pct, screen_keep_within_pct floor, band_max_pct ceiling); use documented dispersion fallback if a matched canary pair is missing. Validate same preconditioning/no-flush execution (including THA27's B2 baseline), quiet/noisy synthetic and real-device comparisons, an unaffected quiet-run screen and improved noisy-run retained-candidate coverage without widening beyond cap. Record canary values, band and shortlist provenance; no candidate admission change until measured hardware/CI regression passes, and no duplicate tuner or independent promotion path.

## Notes

Supersedes: HI130
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi130

CONSOLIDATION 2026-11-22 HIP-AUTOTUNE: receiving THA06,THA27,THA31. Existing scope and acceptance are retained. The original historical source plans and evidence are archived as superseded, not completed. No new kernel, orchestrator, scheduler, patch, GPU result, runtime activation or performance qualification is implied. Changes use the existing owning implementation only.

## Change Log

- 2026-09-09T10:48:25.987562+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:28.943683+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.810110+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.293707+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:26:59.894624+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032737_repaired-four-major-tuning-suc_7897
- 2026-09-10T03:27:37.881650+00:00 (updated-by): Updated: section:ledger-events

## Reviews

- RV4499
- 2026-11-22T10:17:55.311018+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
