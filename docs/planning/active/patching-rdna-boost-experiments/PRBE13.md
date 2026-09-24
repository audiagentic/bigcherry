---
id: PRBE13
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:22.843182+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Evaluate shared-expert output-chain fusion

## Description

TODO, blocked on a source-state precondition. Shared-expert output-chain fusion for an exact SIX-NODE MoE model pattern; candidate kernel shexp_down_gated_q8_0 is a PORT CANDIDATE from the historical RD15 fork lineage (source identities 31eb8e953.../upstream 2f0d3c56..., provenance only) -- no BigCherry patch package exists yet. The shared-expert region must be materialized from PRBE19's reviewed "post-fix source state" -- PRBE19 is a separate plan item supplying a corrected source baseline for this fork lineage; it is a precondition to check first, not a patch this item applies. PRBE05 (Q8_1 cache) is only a dependency if a source audit proves real coupling -- do not assume it.

## Steps

1. Check PRBE19's current state (`mcp__ag-planning__plan_get_item PRBE19`) before any other step -- if PRBE19's post-fix source state has not landed/been reviewed, this item is blocked and should stay pending rather than materializing the shexp_down_gated_q8_0 region from an unreviewed source.
2. Once PRBE19's source state is available: identify the exact six-node model/architecture pattern (grep the target model's real ggml graph construction, e.g. build_moe_ffn-style code for the specific shared-expert architecture RD15's fork targeted) before authoring any rewrite -- do not generalize to "MoE models" broadly.
3. Author a new patches/<order>_rd15_shexp_down_gated_q8_0/ package: PROVENANCE dict citing 31eb8e953.../2f0d3c56... and PRBE19's reviewed commit, Edit() calls anchored to the real b11126 graph-construction site for the six-node pattern.
4. Implement/verify: Q8_0 down-projection semantics, F32/scalar gate shape (nelements==1), residual source wiring, matching K/layout, contiguity, workspace scratch ownership, and graph-capture lifetime safety (the workspace must not be reused/aliased across a capture boundary).
5. Compare fused vs unfused reference for numerical equivalence; author false-positive fallback fixtures (any of the six nodes absent, wrong shape/type, or workspace-ownership conflict must fall back to unfused).
6. Run call-weighted timing only after correctness and workspace-safety gates pass, on a real model matching the six-node pattern.

## Detailed Solution & Technical Design

This is model-specific graph fusion gated on an exact six-node pattern, not a general expert optimization -- treat any broader match as a bug, not a feature. The candidate kernel comes from historical fork code that must be re-validated against PRBE19's corrected source state rather than the original (potentially stale/buggy, per the same class of porting bug PRBE14/RD17 already found once) fork snapshot.

## Code Samples & Guidance

No verified b11126 anchor available yet for this exact six-node pattern -- PRBE19's post-fix source state must be read first (step 1) before any real anchor can be cited; do not invent one. Once available, follow the same patch.toml/patch.py convention as sibling patches (schema=1, id="12xx_rd15_shexp_down_gated_q8_0", state="untested", kind="enhancement", backend="hip", experiment-contract="RD15-SHEXP-DOWN-GATED-Q8_0", requires=[] unless PRBE05 coupling is proven, validation-architectures=["gfx1100","gfx1201","gfx1030"]).

## Files

TBD pending PRBE19 -- expected: real MoE shared-expert graph-construction site in llama.cpp's model-building code (not yet located); new patches/12xx_rd15_shexp_down_gated_q8_0/{patch.toml,patch.py}; model/type/scalar/residual fixtures; workspace/capture tests.

## Validation

Offline (once the package exists): `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay 12xx_rd15_shexp_down_gated_q8_0 --source bigcherry-tuning`. Hardware (Brutus, not run here): six-node pattern positive/negative fixtures; numerical equivalence fused vs unfused; workspace-ownership/graph-capture safety; model-applicability check (only the exact target architecture selects); call-weighted timing.

## Effort & Risk

L effort, advanced skill, hard-blocked on PRBE19 -- do not attempt to shortcut by using the original unreviewed RD15 fork source given PRBE14/RD17 already demonstrated this fork lineage can carry lost-precondition porting bugs.

## Standards

Exact pattern; workspace ownership; model-gated promotion; no hidden dependency.

## Acceptance Criteria

Only exact model/graph patterns select; fused output and capture match reference; workspace is safe; nonqualifying paths fall back; promotion requires positive causal evidence and declared PRBE05 relationship.

## Notes

Supersedes: RD15
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd15

Supersedes: RD15 (closed historical predecessor). Preserve source identities 31eb8e953... / upstream 2f0d3c56... in provenance. Use PRBE13 and PRBE19 for live work; RD15/RD25 remain historical references only.

2026-09-24 relevance at b11126: TODO, blocked on PRBE19's post-fix source state (checked, not yet verified landed within this batch's scope -- PRBE19 exists as its own plan item at docs/planning/active/patching-rdna-boost-experiments/PRBE19.md). GPT design request submitted (req_f7861b0b41c84da5, batched with PRBE14); gateway congested at submission -- authored directly, deliberately declining to invent unverified anchors for the six-node pattern pending PRBE19.

## Change Log

- 2026-09-09T10:54:22.843182+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:31.072998+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.188436+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.888350+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:47:47.250628+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024759_three-rdna-fusion-successors-n_3469
- 2026-09-10T02:47:59.479294+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T09:52:19.850215+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:notes
- chg_20260912_095506_cleaned-the-active-rdna-boost_4906
- 2026-09-12T09:55:06.224060+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:35:24.747448+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
