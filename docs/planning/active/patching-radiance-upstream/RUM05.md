---
id: RUM05
order: 5
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:20.204242+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P1
work: L
---

# Unify placement plans and helper reservations with upstream resource accounting

## Description

Implement the target-version portion through RHP13: one immutable owner/local-slot/encoding plan, local placement views and external helper reservations in upstream accounting. Reuse RHP04/RHP09 calibration; do not implement a second allocator or a new optimizer for the initial bump.

## Steps

- Use RHP13's existing unified-resource scope and RHP04/RHP09's calibrated policy as the owner. Define one typed immutable placement generation with engine ranks, expert owner/local slot/encoding and its memory terms.
- Validate native tp/cards against exact ordered engine/helper assignments; upstream cards is an allowed fleet set, not guaranteed rank order. Resolve stable physical identities before allocation.
- Make planner skip-owned, architecture declare, helper setup and observers read views of the same plan. Remove per-step map construction and duplicated JSON/slot-map copies only after auditing lifetimes.
- Extend upstream fleet reservation/grant path at the smallest external-reservation seam for helper host/pinned/VRAM demand. Include queue scratch and the enlarged staging ring, and never subtract one helper allocation twice.
- Keep actual allocation in upstream fleet/device pools. Reuse S8/S10 measured admission for the resolved ceilings; do not bolt on another allocator or a new global solver.
- Keep initial ownership fixed through each parent step. Existing live migration may remain supported only through its proven epoch protocol; no new adaptive ownership during this bump.
- Validate helper coverage for prefill/verify/fallback modes. If skip-owned cannot serve an enabled pass, refuse that combination before start instead of falling back to absent weights.

## Detailed Solution & Technical Design

Plan selection, reservation and execution are separate operations. An admitted plan chooses ownership and typed layout; the fleet allocator grants physical memory; the remote adapter executes a frozen generation. A dense owner/local-slot view indexed by layer/expert avoids repeated hash lookups but must carry exact encoding/geometry and lifetime. Reuse current owner-map digest and mailbox epoch constructs, not a second generation protocol.

Inspect upstream Fleet::grant, load_op, load_host_room and its card accounting before extending. Helper cards may be outside the engine-only native cards set, but all physical device claims must be reconciled in one ledger. Native reservations cover local experts/KV/static arena and external claims cover actual helper demand; define which bytes are shared versus exclusive and give each term one owner.

The first target port should produce the same ownership as the current admitted configuration. Smarter placement later evaluates a bounded candidate set using qualified throughput/link costs and the named service objective, through RHP04/RHP09. Do not introduce a generic optimizer, automatic per-token movements or hard-coded hysteresis constants to migrate a pin.

Upstream model flex may reclaim resources; external helper ownership cannot be silently reclaimed while tickets still reference it. During single-model acceptance reserve the required helper footprint or reject unsupported reclaim. Full fleet transfer/reclaim belongs in RUM10.

## Code Samples & Guidance

Proposed view contract (illustrative; reuse existing types where they already express it):
```cpp
struct ExpertPlacement {
    uint32_t owner_role;
    uint32_t local_slot;
    EncodingId encoding;
};
struct PlacementView {
    Generation generation;
    std::span<const ExpertPlacement> experts;
};
```
Publish one owning plan before declaring kernels. Views stay valid until admissions stop, ranks/helper work drain, and the generation is retired. Do not borrow vectors owned by temporary YAML decode objects.

## Files

- engines/radiance/components/serving/v2/s8_rules.cpp
- engines/radiance/components/serving/v2/s10_resolver.cpp
- engines/radiance/components/serving/resources.cpp
- engines/radiance/components/serving/plan_cache.cpp
- engines/radiance/components/moe-remote/heat_publication.h
- engines/radiance/patches/rad0002_moe_remote_handoff/patch.py
- engines/radiance/patches/rad0011_plan_memory/patch.py

## Validation

Small pure fixtures: exact owner-map agreement; local/helper byte sum without double counting; infeasible budget refusal; stale generation/unsupported pass refusal; single helper reservation release after drain. RUM09 supplies the one active-layout memory/correctness canary. Do not run a layout optimizer campaign.

## Effort & Risk

L: helper bytes can be omitted or double counted, and skip-owned can leave an enabled pass without weights. Use explicit accounting ownership and freeze the selected current policy for initial migration.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

One placement generation and one physical accounting model feed all readers. Queue/staging/helper/KV costs are explicit. Initial target ownership matches the admitted control, and unsupported skip-owned/flex combinations refuse before executing.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM02, RUM03, RUM04. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisites RUM02/RUM03/RUM04. RHP13 remains unified-allocator scope owner and RHP04/RHP09 remain optimization/calibration owners. No second allocator or new solver is part of this migration.

## Change Log

- 2026-11-18T01:28:20.204242+00:00 (created-by): Created by codex
- 2026-11-18T01:37:43.826131+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:43:10.971610+00:00 (updated-by): Updated: section:effort_risk
