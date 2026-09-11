---
id: PPS03
order: 0
plan: patching-patch-system
state: pending
created-at: '2026-09-11T04:52:15.819930+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: M
priority: P2
---

# Split framework/upstream-fixes patch-sets and tighten group/kind taxonomy

## Description

The current `[patch-set.framework]` in config/recipes.toml (15 patches) silently bundles two different kinds of patch: 14 genuine BigCherry-authored core/dispatch/measurement plumbing (`group = "core"`, `kind = "framework"`, `origin = "local"`, permanent) and 1 upstream correctness backport (`1000_rdna4_mmq_q2k_q6k_fix`, `group = "upstream-fixes"`, `kind = "upstream-backport"`, carries `upstream-ref`/`retirement` fields marking it as temporary until PR #25940 lands upstream). These are semantically different things -- one is permanent BigCherry-owned scaffolding, the other is a stopgap for someone else's bug -- and bundling them in one patch-set with one `required-state` obscures that.

Separately, the `group`/`kind` metadata fields themselves have drifted from what they're actually used for. A full catalog survey (2026-09-11) found: 28 patches carry `group = "core"`, but only 14 of those are in `[patch-set.framework]` -- the other 14 (mostly HI67/HI18/HI85/HI119/HI81/HI14/HI134-series correctness/diagnostic/test tooling) are labeled "core" but selected nowhere, so the tag and the shipped set have silently diverged. Additionally one patch is tagged `kind = "upstream-backport"` with `group = "rdna-boosts"` instead of the otherwise-consistent `group = "upstream-fixes"` pairing every other upstream-backport patch uses -- the same class of mislabeling as the framework/upstream-fixes bundling issue above, just at the individual-patch level.

Full group/kind combination census across all 68 patches: 28x (core, framework), 19x (rdna-boosts, enhancement), 8x (nasone-rdna, enhancement), 6x (upstream-fixes, upstream-backport), 2x (gpu-collectives, enhancement), 1x (rdna-boosts, upstream-backport) [the mismatch], 1x (_template, unset).

## Steps

1. Add a new `[patch-set.upstream-fixes]` in config/recipes.toml (or equivalent composition point) and move `1000_rdna4_mmq_q2k_q6k_fix` into it, out of `[patch-set.framework]`. Confirm `[source.bigcherry]`'s composition still includes both sets for a real build (this must not silently drop the fix from production).
2. Audit the 14 `group = "core"` patches that are NOT in `[patch-set.framework]` (0840_hybrid_allreduce_dispatch, 0850_ordered_speculative_trace, and the HI67/HI18/HI85/HI81/HI14/HI105/HI119(x3)/HI134 series) -- for each, decide and record: relabel to a more accurate group (they look like correctness-probe/diagnostic/test tooling, not core framework plumbing), or genuinely belong in `[patch-set.framework]` and were just missed.
3. Fix the single `(rdna-boosts, upstream-backport)` mismatched patch to `group = "upstream-fixes"` to match every other upstream-backport patch's pairing.
4. Add a `patch-lint` rule (or extend cross_check) that fails closed on: (a) group/kind pairing inconsistency against the established combinations, (b) any `kind = "upstream-backport"` patch missing `upstream-ref` or `retirement` fields, (c) optionally, a `group = "core"` patch not present in any patch-set (to catch future drift like finding #2 above before it accumulates again).
5. Re-run patch-lint, check, and the full offline test suite after all relabeling to confirm nothing regresses.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

config/recipes.toml's framework patch-set contains only group=core/kind=framework patches; upstream-fixes patches are tracked in their own set and still compose into a real production build; every group=core patch is either in a patch-set or has its group corrected to reflect its actual role; the (rdna-boosts, upstream-backport) mismatch is fixed; a patch-lint rule enforces group/kind consistency going forward; full test suite passes.

## Notes

This is a metadata/taxonomy cleanup, not a behavior change to any patch's actual transform -- do not touch patch.py anchors as part of this item. The goal is that `group`/`kind` mean what they claim and patch-set membership is fully explainable from that metadata, not that new fields/tags need to be invented -- the existing vocabulary (core/upstream-fixes/rdna-boosts/nasone-rdna/gpu-collectives x framework/upstream-backport/enhancement) already covers everything found; it just hasn't been enforced or fully applied.

Raised during a 2026-09-11 user-led survey of "which patches are validated vs which are core vs optimizations" that surfaced the framework/upstream-fixes bundling as a real, unprompted finding (not something either party set out looking for).

## Change Log

- 2026-09-11T04:52:15.819930+00:00 (created-by): Created by agent
- 2026-09-11T04:52:34.707749+00:00 (updated-by): Updated: priority='P2', section:description, section:steps, section:acceptance_criteria, section:notes
