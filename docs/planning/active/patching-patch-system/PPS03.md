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

# Add an enumerated, sparse `tags` field; fix kind mislabels and framework/upstream-fix composition bug

## Description

Design settled through discussion (2026-09-11), superseding earlier drafts of this item:

**`kind` stays the primary, small, fixed field builds key off**: `framework` / `diagnostic` / `upstream-backport` / `enhancement`. This is what patch-set composition actually selects on.

**`group` is dropped**, not collapsed. Once provenance (`origin`+`external-source`, already exist) and subsystem (`subsystems`, already exists) are recognized as the fields that actually carry that information, `group` had nothing left to encode that wasn't already redundant with `kind` -- keeping two fields in lockstep for the same information was worse than one. Real confirming case: `1234_rd58_pin_state_buffer_multigpu_restore` is `group="rdna-boosts"` + `kind="upstream-backport"` -- a legitimate external-PR backport (external-source="davetha-llama-cpp") that is part of the rdna-boosts experimental initiative. This is not a mislabel; it shows a single named initiative can legitimately contain patches of more than one kind, which is exactly why a strict group<->kind bijection (an earlier draft of this item) was the wrong model.

**New: an enumerated, sparse, multi-valued `tags` list field**, for BigCherry's own cross-cutting classification/analysis -- not a build-selection input. Applied only where genuinely true and specific, never exhaustively: an architecture-independent optimization gets no architecture tag; something genuinely RDNA4-only gets `rdna4` and nothing else. The enum is expected to grow as new patch types appear, same as any other living vocabulary -- this is a starting point derived from the real patch set (2026-09-11 survey), not a closed list.

Initial vocabulary, derived from real data already in the catalog (most of these fields are currently populated on only 4-8 of 68 patches, confirming this needs real backfill, not just a schema addition):
- Purpose: `fix`, `optimization`, `enhancement`, `tuning`, `diagnostic`, `framework`
- Architecture: `gfx1100`, `gfx1101`, `gfx1151`, `gfx1201`, `rdna3`, `rdna3.5`, `rdna4`, `dual-gpu`, `peer-access`
- Subsystem: `allreduce`, `tensor-parallel`, `p2p`, `meta-backend`, `graph-fusion`, `gated-delta-net`, `mtp`, `state-snapshots`, `wmma`, `prefill`, `speculative-decoding`, `top-k`, `moe-routing`, `wave32`, `flash-attention`, `mmvq`, `mmq`, `mmvf`, `quantization`, `kv-cache`, `dispatch`

Two real, narrower bugs found during the survey remain in scope (neither is a group/kind labeling problem after this redesign):
1. 14 patches (0840_hybrid_allreduce_dispatch, 0850_ordered_speculative_trace, and the HI67/HI18/HI85/HI81/HI14/HI105/HI119(x3)/HI134 series) are currently `group="core"` -- under the new scheme they should be `kind="diagnostic"`, not `kind="framework"` (they are correctness-probe/test/tracing tooling, not build/dispatch plumbing).
2. `1000_rdna4_mmq_q2k_q6k_fix` is correctly tagged `kind="upstream-backport"` but is nonetheless listed in `[patch-set.framework]` in config/recipes.toml -- the composition doesn't match its own kind. This is a recipes.toml bug, not a patch.toml mislabel.

## Steps

1. Remove the `group` field from patches/_template/patch.toml and the patch schema/validation code that reads it; migrate every patch's existing `group` value out (most map directly to the `kind` they already had, since group and kind were redundant for the vast majority of patches -- only the rdna-boosts/nasone-rdna/gpu-collectives-labeled ones need review to confirm their `kind` value is independently correct, not just inherited from the removed group).
2. Add a `tags` list field to the patch schema (empty list default, no requirement to populate). Add the enumerated vocabulary above as the initial allowed set in whatever validates patch.toml (patch_catalog.py / patch_registry.py) -- fail closed on an unrecognized tag, but never require any tags be present.
3. Fix the 14 misfiled patches' `kind` from `framework` to `diagnostic` (see bug #1 above).
4. Fix config/recipes.toml: create `[patch-set.upstream-fixes]`, move `1000_rdna4_mmq_q2k_q6k_fix` out of `[patch-set.framework]` into it, and add `upstream-fixes` to the patch-sets of BOTH `[source.bigcherry-native]` and `[source.bigcherry]` (a correctness fix should never be optional the way an enhancement is -- see bug #2 above).
5. Backfill `tags` on at least the patches already touched by this item (the 14 diagnostic ones, the upstream-fixes ones, and any patch whose title/content makes an architecture or subsystem tag unambiguous) as a real starting population, not a placeholder.
6. Add/extend a patch-lint rule that fails closed on any `tags` entry outside the enumerated vocabulary, and confirms `kind="upstream-backport"` patches carry `upstream-ref`+`retirement`.
7. Re-run patch-lint, check, and the full offline test suite; confirm both `bigcherry-native` and `bigcherry` still compose correctly with `1000_rdna4_mmq_q2k_q6k_fix` reachable via upstream-fixes in both.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

`group` field removed from the patch schema and every patch.toml. New `tags` list field exists, is optional/sparse, and is validated against an enumerated vocabulary (fail closed on unknown tags, never required to be non-empty). The 14 misfiled diagnostic patches carry `kind="diagnostic"`. `1000_rdna4_mmq_q2k_q6k_fix` is composed via a new `upstream-fixes` patch-set present in both bigcherry-native and bigcherry sources, not `[patch-set.framework]`. patch-lint enforces the tag vocabulary and upstream-backport field requirements. Full test suite passes.

## Notes

This is a metadata/taxonomy cleanup, not a behavior change to any patch's actual transform -- do not touch patch.py anchors as part of this item. The goal is that `group`/`kind` mean what they claim and patch-set membership is fully explainable from that metadata, not that new fields/tags need to be invented -- the existing vocabulary (core/upstream-fixes/rdna-boosts/nasone-rdna/gpu-collectives x framework/upstream-backport/enhancement) already covers everything found; it just hasn't been enforced or fully applied.

Raised during a 2026-09-11 user-led survey of "which patches are validated vs which are core vs optimizations" that surfaced the framework/upstream-fixes bundling as a real, unprompted finding (not something either party set out looking for).

This is a metadata/taxonomy change, not a behavior change to any patch's actual transform -- do not touch patch.py anchors as part of this item.

Design history for context: an early draft proposed collapsing `group` to 4 values matching `kind` 1:1 (rejected as pure redundancy once spotted); a middle draft proposed keeping `group` as an independent, kind-crossing initiative label (rejected once it became clear provenance/subsystem already cover that need via existing fields, making a THIRD field for the same purpose unnecessary). The final design removes `group` and adds `tags` as a genuinely new, sparse, non-required axis that doesn't duplicate `kind`, `origin`/`external-source`, or `subsystems`.

Raised during a 2026-09-11 user-led survey of "which patches are validated vs core vs optimizations" that surfaced all of the above as real, unprompted findings.

## Change Log

- 2026-09-11T04:52:15.819930+00:00 (created-by): Created by agent
- 2026-09-11T04:52:34.707749+00:00 (updated-by): Updated: priority='P2', section:description, section:steps, section:acceptance_criteria, section:notes
- 2026-09-11T04:54:16.836438+00:00 (updated-by): Updated: section:steps
- 2026-09-11T05:04:26.715764+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:acceptance_criteria, section:notes
