---
id: PRBE72
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:35.919608+00:00'
breadth: ''
skill: ''
created-by: capability-rebaseline-v3
priority: null
---

# Hardware-eligibility needs device traits beyond architecture (integrated/UMA/P2P/driver/GPU-count)

## Description

OBSOLETE (already implemented). tools/bigcherry/experiment/contract.py's `Scope` dataclass (line ~576) already carries `integrated: bool | None`, `uma: bool | None`, `peer_access: bool | None`, `gpu_count: GpuCountConstraint | None`, and `driver: DriverVersionConstraint | None` alongside `architectures`, and `evaluate_scope_eligibility()` already fails closed on missing hardware observations for trait-bearing scopes. The dataclass's own docstring cites RD22 by name as the motivating case, already expressible as `Scope(backend="hip", architectures=("gfx1151",), integrated=True, uma=True)`.

TODO (reopened, GPT WRONG-confirmed). The Scope schema half is genuinely implemented: tools/bigcherry/experiment/contract.py's Scope dataclass carries integrated/uma/peer_access/gpu_count/driver fields and evaluate_scope_eligibility() fails closed on missing observations for trait-bearing scopes. But the item's own required consequence -- migrating RD22 to use those traits -- is NOT done: config/experiment-contracts.toml line ~488 still declares `[[...]] title = "RD22: back out integrated-GPU host buffers on HIP..."` with `architectures = ["gfx1201"]` and no `integrated`/`uma` fields, even though Scope's own docstring names RD22 as the motivating case for expressing it as gfx1151+integrated+UMA. The schema existing does not make the item OBSOLETE while its own named consumer hasn't migrated.

## Steps

Add device-trait fields integrated, uma, peer_access, gpu_count and driver to Experiment Contracts/autotune hardware key; audit RD22/PRBE17 and similar items for architecture-versus-trait eligibility; update contract scopes and tests so traits drive selection where they are the real condition.

1. Re-read config/experiment-contracts.toml around line 472-488 (RD22 contract block) to get its exact current field set.
2. Add `integrated = true` and `uma = true` to the RD22 Scope block (and correct `architectures` to the traits-based gfx1151 scope the Scope docstring describes, if gfx1151 is in fact the target hardware for RD22 -- verify against PRBE17/patch 1209's actual target arch before changing `architectures`, since the current value is gfx1201 and may itself need reconciling rather than blind copy from the docstring).
3. Run the experiment-contract schema validation / contract loader tests to confirm the migrated RD22 block still parses and evaluate_scope_eligibility() behaves as expected for an integrated/UMA host.
4. Audit whether any other hardware-scoped contract in experiment-contracts.toml should be using integrated/uma/peer_access/gpu_count/driver traits instead of architecture-only scoping, and list them in notes as follow-up candidates (do not migrate them here -- keep this item scoped to RD22).

## Detailed Solution & Technical Design

Architecture strings alone misclassify integrated/UMA correctness work. Preserve architecture as one dimension but include topology/device traits and driver identity in eligibility and candidate keys, with backward-compatible defaults and fail-closed unknown traits.

Scope's trait fields are additive and optional (None means unconstrained), so migrating RD22 is a config-only change: add the two boolean trait fields to its existing TOML block. The risk is scope drift -- since RD22's `architectures` currently says gfx1201, not gfx1151, this must be reconciled against PRBE17 (the superseded item this contract binds to) before touching architectures, not assumed from the Scope docstring alone.

## Code Samples & Guidance



## Files

Experiment Contract schema and hardware-key resolution; device discovery; RD22/PRBE17 contract updates; eligibility serialization and tests.

config/experiment-contracts.toml (RD22 block ~472-488, edit); tools/bigcherry/experiment/contract.py (Scope, evaluate_scope_eligibility -- evidence only, already correct).

## Validation

Unit/integration tests for trait combinations, serialization compatibility, integrated versus discrete/P2P and multi-GPU selection.

Offline: TOML parses; any experiment-contract loader/schema unit tests in tests/ covering Scope trait fields; confirm evaluate_scope_eligibility(RD22_scope, observed_traits) returns the expected eligible/ineligible result for an integrated+UMA host vs a discrete host.

## Effort & Risk

S -- config-only change, but requires correctly resolving the gfx1201-vs-gfx1151 architecture question against PRBE17 first.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Hardware-scoped items select by proven device traits rather than architecture alone, with unknown traits falling back safely and existing contracts remaining compatible.

## Notes

Supersedes: RD92
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd92

2026-09-24 relevance: OBSOLETE, premise already implemented. Verified via direct read of tools/bigcherry/experiment/contract.py lines ~560-600 (Scope dataclass with integrated/uma/peer_access/gpu_count/driver fields, evaluate_scope_eligibility fail-closed helper, RD22 named in the docstring as the motivating case). The remaining piece of the item's own ask -- 'audit RD22/PRBE17 and similar items for architecture-vs-trait eligibility' -- may still be open work; PRBE16 (gfx1151 defer item) does not yet reference Scope's trait fields explicitly, so if that audit is still wanted it should be filed as a fresh, narrowly-scoped item (e.g. 'apply Scope trait fields to PRBE16/RD22's own contract declaration') rather than kept alive under this item's now-satisfied schema-design premise. GPT design request: not submitted -- disposition made directly from source inspection (schema already exists, no design question remains); no GPT request id.

2026-09-24 GPT review req_d55aed71224e43a8 applied: WRONG disproven -- Scope schema exists but RD22's own contract block (experiment-contracts.toml ~488) was never migrated to use integrated/uma traits; reopened to pending, scoped to that migration.

## Change Log

- 2026-09-09T10:58:35.919608+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:46.423644+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.454152+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.291596+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:20:20.626577+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032047_repaired-five-more-active-succ_6361
- 2026-09-10T03:20:47.947706+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:35:00.841581+00:00 (updated-by): Updated: section:description, section:notes
- 2026-09-24T02:35:15.360640+00:00 (state-transition): State: pending → deprecated
- 2026-09-24T04:35:32.676165+00:00 (state-transition): State: deprecated → pending
- 2026-09-24T04:37:14.720909+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk
- 2026-09-24T04:37:25.933906+00:00 (updated-by): Updated: section:notes
