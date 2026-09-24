---
id: PRBE42
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:23.562520+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# AMD-GDN-001: Chunked fused GatedDeltaNet recurrence

## Description

IMPLEMENTED-AS-PATCH. Patch 1221_rd50_gdn_chunked_recurrence (state=untested) implements the chunked fused GatedDeltaNet recurrence, folding upstream-fork files gated_delta_net_chunked.cu/.cuh into gated_delta_net.cu, adding a GDN_CHUNK=32-token chunked kernel narrowly gated (scalar gate only, S_v==128, n_tokens>32, RDNA3_5) with an env-var force-disable for A/B benching. This patch SUBSUMES RD51/RD52/RD53 (PRBE43/44/45) per its own SUMMARY.md: DPP reduction, native exp2 decay, and launch-bounds tuning are inline micro-decisions inside the same kernel body in the source PR, not independently portable hunks, so they are one patch, not four. This item's own eligibility gate (scalar gate, K=1, S_v=128, gfx1151) matches the patch's real gating condition.

## Steps

1. Read patches/1221_rd50_gdn_chunked_recurrence/patch.py to confirm its Edit() anchors still apply cleanly at b11126 (SSM/GDN op code may have shifted between b10901 and b11126 -- this patch predates the current pin per its provenance).
2. Run patch-lint and patch-rebase-check to confirm mechanical validity before any hardware run.
3. Real gfx1151 hardware is required to qualify performance (the source's 1.89x/+20% headline claims are explicitly NOT acceptance by themselves per this item's own acceptance criteria) -- schedule via validation_campaign.py, not run here.
4. Confirm real gfx1100/gfx1201 fallback (old token-by-token kernel path) is exercised and unaffected -- correctness-only check, no gfx1151 hardware needed for this half.
5. Recurrent-state/output parity vs the old kernel over long sequences plus PPL/deterministic model checks (Qwen hybrid) must pass before any performance claim is trusted.

## Detailed Solution & Technical Design

Replace token-by-token GDN overhead with a chunked HIP recurrence that keeps state in registers/LDS, but only under the exact supported shape predicate: scalar gate, K=1, S_v=128, target gfx1151/RDNA3.5, and eligible prefill ubatches. The eligibility gate must fail closed on gfx1100/gfx1201, decode n=1, other gate modes, S_v/K values, and unsupported model graphs. Preserve the existing dispatch path and state semantics outside the gate.

## Code Samples & Guidance

Trigger: Qwen hybrid/GDN prefill on gfx1151 with scalar gate, K==1, S_v==128 and ubatch 512..4096. Controls: other S_v/K/gate modes, decode n=1, standard attention models, gfx1100/gfx1201. Boundary: chunk size, context length, workspace, VGPR/LDS pressure.

## Files

patches/1221_rd50_gdn_chunked_recurrence/patch.py, patch.toml, README.md, evidence/ (existing); config/experiment-contracts.toml (check for an RD50-scoped contract; author if missing).

## Validation

Offline: PYTHONPATH=tools python -m bigcherry patch-lint patches/1221_rd50_gdn_chunked_recurrence; patch-rebase-check --focal-overlay 1221_rd50_gdn_chunked_recurrence --source bigcherry-tuning. Hardware (on Brutus, not run here): gfx1100/gfx1201/gfx1030 fallback correctness (old kernel path unaffected); gfx1151 direct GDN op parity (recurrent state/output vs old kernel, long sequences) plus Qwen hybrid PPL/deterministic checks; only after correctness passes, gfx1151 GDN op time + E2E prefill benchmark with VGPR/LDS/workspace reporting and decode n=1 neutrality.

## Effort & Risk

L; invasive kernel and state changes are model-specific. Hardware performance remains blocked without gfx1151, so separate compile/fallback evidence from performance qualification.

## Standards

Fail-closed hardware/shape gating, preserve recurrent state semantics, no unsupported fallback regression, and retain campaign evidence/provenance.

## Acceptance Criteria

Acceptance requires exact scalar-gate/K=1/S_v=128 gfx1151 eligibility, long-sequence recurrent-state/output parity, verified gfx1100/gfx1201 and unsupported-shape fallback, and repeatable GDN/E2E prefill benefit without decode regression; source headline claims alone are insufficient.

## Notes

Supersedes: RD50
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd50

Supersedes RD50. Root prerequisite for PRBE43 and the later GDN-003/GDN-004 successors. Existing compile-safety and fallback evidence does not substitute for gfx1151 performance qualification.

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH (1221_rd50_gdn_chunked_recurrence, state=untested; found via `grep -rl RD50 patches/*/patch.toml`-equivalent search, SUMMARY.md confirms plan item RD50/RD51/RD52/RD53 mapping). Patch already subsumes PRBE43/44/45 by design (single kernel body, not separable hunks) -- see those items' notes for the same disposition. No GPT design session needed (patch already exists and matches this item's own gating description); qualification is real-hardware evidence gathering, not new design. GPT gateway was busy earlier this session (see PRBE32 notes) but was not needed for this item.

## Change Log

- 2026-09-09T10:56:23.562520+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:38.267576+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.319945+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.085474+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:04:32.318831+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- chg_20260910_030451_carried-forward-the-detailed-s_2071
- 2026-09-10T03:04:51.342920+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:05:58.216876+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030619_removed-migration-placeholder_7703
- 2026-09-10T03:06:19.338602+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:33:21.468047+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation
- 2026-09-24T02:33:26.697599+00:00 (updated-by): Updated: section:notes
- chg_20260924_023553_re-scoped-11-rdna-boost-planni_1625
- 2026-09-24T02:36:27.926898+00:00 (updated-by): Updated: section:ledger-events
