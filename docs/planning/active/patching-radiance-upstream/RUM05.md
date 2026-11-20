---
id: RUM05
order: 5
plan: patching-radiance-upstream
state: in_progress
created-at: '2026-11-18T01:28:20.204242+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P1
work: L
---

# Preserve one ownership plan and disjoint-card admission without a new fleet broker

## Description

Align the initial single-model migration with upstream explicit card order and existing BigCherry measured admission. Share immutable ownership views and remove repeated mapping work; defer broker/shared-card extensions.

## Steps

- Freeze one production model with explicit engine card list and disjoint helper cards; validate tp, identities and no overlaps before starting processes.
- Translate admitted physical BDFs through current ROCR/HIP visibility masking to the actual child-local ordinal namespace. Preserve ordering across native YAML serving and offline measuring/KL Config.
- Reuse upstream single-model Fleet::choose_cards order; add only a bounded mixed-ISA eligibility change if Engine::usable_cards incorrectly rejects a needed production device.
- Share admitted owner/local-slot/encoding views with planner, helper setup and runtime. Preserve the helper's independent bc::experts::plan_owners verification; recomputation there is an integrity check, not redundant policy.
- Keep measured per-BDF host/pinned/VRAM/helper budgets and existing skip-owned checks, incorporating changed queue/staging costs.
- Do not add Fleet::grant/LoadBroker hooks for this single-model path: use_broker is false. Leave overlapping cards, flex lending and shared reclaim to RUM10.
- Keep current calibrated ownership policy for acceptance; later optimization remains RHP04/RHP09/RHP13.

## Detailed Solution & Technical Design

Upstream fleet.cpp use_broker() returns models_.size()>1. For one model choose_cards assigns the first tp entries of cards_, run preserves explicit list order, and load_op writes cfg.devices. Native offline main creates Engine directly and does not transfer fc.cards; carry the same validated selected vector there explicitly.

Current session launch masks visible devices, so host GPU ordinals and engine child-local ordinals differ. Identity/admission must resolve physical cards once and construct both the mask and native local indices from the same ordered selection. Do not compare host ordinals directly to YAML child indices.

No new allocator is necessary for disjoint cards: retain current per-device measured admission, native engine allocation and helper footprint checks. Full shared fleet reservations require a different integration and remain optional later work. Removing plan recomputation at worker startup would weaken independent owner-map verification; keep it.

## Code Samples & Guidance

Verified native path: Fleet::run explicit cards order -> choose_cards first tp when !use_broker -> load_op cfg.devices.

Required local mapping:
```text
ordered admitted BDFs -> launcher visibility mask -> child-local cfg.devices
```
Use the same selection for offline Engine construction. Gate wrong-order, masked namespace and helper-overlap refusals with small fixtures.

## Files

- engines/radiance/components/serving/v2/s8_rules.cpp
- engines/radiance/components/serving/v2/s10_resolver.cpp
- engines/radiance/components/serving/resources.cpp
- engines/radiance/components/serving/plan_cache.cpp
- engines/radiance/components/moe-remote/heat_publication.h
- engines/radiance/patches/rad0002_moe_remote_handoff/patch.py
- engines/radiance/patches/rad0011_plan_memory/patch.py

## Validation

Reuse small fixtures for exact ordered BDF to visible mask to child-local device binding, including a nontrivial normal/offline order; local/helper byte sums; infeasible fresh budget; independent worker owner verification; unsupported-pass refusal and existing helper reservation release after drain. RUM09 supplies active-layout memory/correctness. No new broker or optimizer campaign.

## Effort & Risk

L: helper bytes can be omitted or double counted, and skip-owned can leave an enabled pass without weights. Use explicit accounting ownership and freeze the selected current policy for initial migration.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

The admitted single-model engine/helper assignment matches exact physical identities and child-local rank order in serving and offline mode. Existing measured admission and independent ownership checks remain. No broker hook or competing allocator is added for this scope.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM02, RUM03, RUM04. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisites RUM02/RUM03/RUM04. RHP13 remains unified-allocator scope owner and RHP04/RHP09 remain optimization/calibration owners. No second allocator or new solver is part of this migration.

Astra source review ASTRA-CONSOLIDATION.md supersedes the earlier mandatory Fleet::grant external-reservation proposal for the initial bump; dependencies RUM02/RUM03/RUM04. Shared-card integration is RUM10 only.

Native owner now repeats the shared S8 host/cgroup/pin room and ordered BDF gates after power admission, before any serving child forks. The new gate uses authenticated helper RSS and the sealed manifest; existing VRAM checks remain. Final v3 owner and S8 tests passed 2/2. Focused v2 serving, session, stored and S8 tests passed 4/4. Leased physical placement and offline rank-order qualification remain open.

2026-11-20 defect from the RUM09 per-config smokes: flashnext-r9700 (single card, no helpers) is refused at the engine's live admission: vram on 0000:09:00.0 live 32999625720 bytes short of required 33000662008 (shortfall 1036288 bytes; statics 5800083456, experts 26852451320, kv_required 348127232, headroom 0, prefill chunks 64). The plan sizes the elastic budget to the exact claimable bytes with zero headroom, and 2.2 holds slightly more at serve time (driver context 329 vs 258 MiB and queue scratch observed on flashnext-4gpu). Fix in the planner: derive the serve-time residency from the same measured inputs the engine admits against (or remeasure at admission and re-plan), not a hand-picked margin. Also fixed on the branch for mixed-arch worlds: rank-devices omitted from the YAML takes the admitted BDF order (64bae9124).

2026-11-21 second admission defect from the RUM09 smokes: flashnext-2xtx, flashnext-2xtx-combine and flashnext-2xtx-rdna2-head stop at engine start with '--ngram-placement ram: the n-gram table (47.68 GiB) cannot be held in memory as a pinned copy': host 91 GiB, available about 76-82 GiB, 27-30 GiB more still to allocate (pinned host pools 11-17 GiB for two-card helper layouts, prefix-cache host tier, checkpoint slots, host activation arenas). The configs inherit ngram_placement=ram from the flashnext base; v1.17 could fall back, but 2.2 admission requires an explicit placement and refuses auto. Fix in the planner, not per config: derive the n-gram placement (ram or disk) from the measured host room and the other host allocations, record it explicitly in the plan and manifest, and refuse with both numbers only when neither fits. Same principle as the flashnext-r9700 VRAM case above: one measured derivation, no hand-picked margins.

## Change Log

- 2026-11-18T01:28:20.204242+00:00 (created-by): Created by codex
- 2026-11-18T01:37:43.826131+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:43:10.971610+00:00 (updated-by): Updated: section:effort_risk
- 2026-11-18T01:56:51.018833+00:00 (updated-by): Updated: section:code_samples
- 2026-11-18T02:16:44.473099+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:detailed_solution, section:code_samples, section:acceptance_criteria, section:notes
- 2026-11-18T02:23:07.396066+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:26:08.989224+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:28:17.614303+00:00 (updated-by): Updated: section:validation
- 2026-11-19T22:18:39.878379+00:00 (state-transition): State: pending → in_progress
- 2026-11-19T22:59:23.833156+00:00 (updated-by): Updated: section:notes
- 2026-11-20T12:44:34.149699+00:00 (updated-by): Updated: section:notes
- 2026-11-20T13:12:22.293227+00:00 (updated-by): Updated: section:notes
