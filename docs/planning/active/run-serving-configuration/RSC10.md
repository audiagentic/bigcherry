---
id: RSC10
order: 0
plan: run-serving-configuration
state: pending
created-at: '2026-11-05T18:06:22.609430+00:00'
breadth: ''
skill: advanced
created-by: bigcherry-2
priority: P1
work: M
---

# Host ledger margin refit: a start that fails the engine's host-room check stores its shortfall and retries once

## Description

The host ledger leaves a calibrated margin unplanned (calibration.host_ledger_margin.mib, a seed: 1024 MiB, with its evidence in calibration.toml) for what the host loses between the manifest being written and the engine's own host-room check. The seed is a judgement (about 3.6 x the largest observed unexplained loss in the RSC08 S3 hardware gate), not a derivation, and a plan filled too tightly stops the engine at start (observed: 205 MiB short at 75 MiB spare). This item replaces the seed by measurement: a start that fails the engine's own host-room check records the observed shortfall, and the next manifest carries a margin raised to it, so a miss costs one cheap retry, not a human.

## Steps

1. The engine's host-room refusal (the n-gram, embedding and tower checks in engine_bringup.cpp) is reported through a structured, stable exit (not text) with the figures it used: MemAvailable, the allocations still to come, the headroom.
2. The launcher, on that exit before any weight loaded, computes the shortfall (needed minus available), stores it with the plan-cache entry as a measured quantity with provenance, writes a new manifest whose calibration cache entry (source cache) raises host_ledger_margin.mib to cover it (observed shortfall plus the margin already planned), and retries the start once. A second failure stops with both figures named.
3. Every start records the engine's actual host-room figure (available, needed, spare) beside the plan-cache entry, so the margin can be refit downward as well as upward from real starts (the refit keeps the largest shortfall seen in a window, never a single sample).
4. The seed stays as the floor on an unseen layout and is labelled provenance seed in spec.calibration; a cache entry replaces it on the same start that measured it.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Card-free: a fake engine that refuses with a shortfall gives one retry with the raised margin, then the start; a second refusal stops with both figures; the retry's manifest differs from the first only in resources.host.checkpoint_slots, spec.calibration and the digest, and the plan key does not move (already tested in the writer for a calibration cache refit). Hardware: a start deliberately planned to 0 MiB spare (margin 0 in a cache entry) refuses at the engine's check, retries once with the measured shortfall, and serves; the 6-stream soak passes 3 times on the slot count it picks.

## Effort & Risk

M. Risk: the engine's refusal text is the only signal today; the structured exit is an engine patch (rad0011 or a new one) and a pin-bump rebase cost.

## Standards

arch-standards

## Acceptance Criteria

- A host-room refusal at start is never a human task: one automatic retry with a measured margin, or a refusal naming both figures.
- The margin in force is always visible in spec.calibration with its source (seed or cache) and evidence.
- No hand-picked constant remains except the labelled seed floor.

## Notes



## Change Log

- 2026-11-05T18:06:22.609430+00:00 (created-by): Created by bigcherry-2
- 2026-11-05T18:06:44.484861+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:standards, section:acceptance_criteria
