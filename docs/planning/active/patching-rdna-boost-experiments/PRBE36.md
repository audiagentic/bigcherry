---
id: PRBE36
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:57.373909+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-STREAM-006: Enable graph-opt by default on RDNA3.5

## Description

IMPLEMENTED-AS-PATCH. Patch 1217_rd44_graph_opt_default_rdna35 (state=untested) defaults GGML_CUDA_GRAPH_OPT to enabled on gfx1151 (GGML_CUDA_CC_IS_RDNA3_5) only, leaving every other architecture's env-gated-off default unchanged, with GGML_CUDA_GRAPH_OPT remaining an explicit override in both directions. Depends on patches 1215/1216 (RD39-43) for correctness once triggered -- this item's own acceptance criteria explicitly block default-on until PRBE35 (1216's qualification) passes. Real per-architecture caveat found by PRBE35's own evidence: gfx1100 shows a clear positive gain (+2.38% mean, 10-round CI95), gfx1201 is noisier/weaker, gfx1030 shows a net-negative/inconclusive signal -- 1217 currently only touches gfx1151 (RDNA3.5) default, which none of that evidence directly covers (the gathered evidence is gfx1100/gfx1201/gfx1030, not gfx1151/RDNA3.5). This item therefore still needs its own dedicated gfx1151 regression evidence before promotion; it is not satisfied by PRBE34/35's gfx1100-focused campaign.

## Steps

1. Do not promote until PRBE35 (1216 qualification) reaches a formal eligible_for_validated_state verdict -- hard blocking dependency per this item's own acceptance criteria.
2. On real gfx1151 hardware (not available in this planning session), run the broad dense/MoE/GDN/MTP on/off regression suite this item's validation section already specifies: capture stress, long-context edge cases, correctness/output divergence, and performance distribution across multiple models -- not a single-model extrapolation from the gfx1100 evidence gathered under PRBE34/35.
3. Confirm non-gfx1151 architectures (gfx1100/gfx1201/gfx1030) are unaffected by 1217's default change (it should be a pure no-op there since the env var default stays off) via a quick smoke build/run control.
4. If gfx1151 hardware-wide confidence is absent (e.g. only theoretical/no real device access), retain 1217 as opt-in (do not promote) and record that as the blocking evidence explicitly, per this item's own acceptance criteria -- this is an acceptable, documented outcome, not a stalled item.

1. Do not promote until PRBE35 (1216 qualification) reaches a formal eligible_for_validated_state verdict -- hard blocking dependency.
2. Add `experiment-contract = "RD44-GRAPH-OPT-DEFAULT-RDNA35"` to patches/1217_rd44_graph_opt_default_rdna35/patch.toml (verified: currently requires 1215+1216 correctly but has no experiment-contract binding field). Author the RD44-GRAPH-OPT-DEFAULT-RDNA35 contract in config/experiment-contracts.toml if it does not already exist, following the RD39-42/RD43 pattern: subject=env unset/default-on, control=explicit GGML_CUDA_GRAPH_OPT=0, target architecture gfx1151.
3. Add patches/1217_rd44_graph_opt_default_rdna35/validation.toml and a patch-local validation/producer.py emitting backend_reference correctness, activation, and paired performance/control regression evidence for gfx1151 specifically.
4. Run the gfx1151 qualification (broad dense/MoE/GDN/MTP on/off regression suite, capture stress, long-context edge cases) on an actual gfx1151 host -- Brutus does NOT have a gfx1151 device (its GPUs are gfx1100/gfx1201 per this project's dual-XTX baseline); use Brutus only for the non-target gfx1100/gfx1201/gfx1030 no-op smoke control in step 5, never for the gfx1151 qualification itself.
5. Confirm non-gfx1151 architectures (gfx1100/gfx1201/gfx1030) are unaffected by 1217's default change via a quick smoke build/run control on Brutus (correct host for this control only).
6. If gfx1151 hardware access remains unavailable, retain 1217 as opt-in and record that as the explicit blocking evidence -- an acceptable, documented outcome.

## Detailed Solution & Technical Design

No design work is needed here: 1217's mechanism (a bool default keyed off GGML_CUDA_CC_IS_RDNA3_5(cc), verified as a real upstream macro at b11126 ggml/src/ggml-cuda/common.cuh:91) is already implemented and its own acceptance criteria are already precisely the gating logic this item asks for. The remaining work is entirely evidence-gathering (gfx1151 hardware campaign) and dependency-sequencing (wait for PRBE35), not further code design.

## Code Samples & Guidance

No new code_samples needed; 1217's patch.py already contains the implementation (read patches/1217_rd44_graph_opt_default_rdna35/patch.py for the exact anchor before any qualification run).

## Files

patches/1217_rd44_graph_opt_default_rdna35/patch.py (existing, no change expected); patches/1217_rd44_graph_opt_default_rdna35/README.md (add the gfx1151 regression evidence once gathered); config/experiment-contracts.toml (check for an RD44-scoped contract; author one if missing, following the RD39-42/RD43 pattern).

patches/1217_rd44_graph_opt_default_rdna35/patch.py (existing, no change expected); patches/1217_rd44_graph_opt_default_rdna35/patch.toml (add experiment-contract binding); patches/1217_rd44_graph_opt_default_rdna35/validation.toml (new); patches/1217_rd44_graph_opt_default_rdna35/validation/producer.py (new); config/experiment-contracts.toml ([contract.RD44-GRAPH-OPT-DEFAULT-RDNA35], author if missing).

## Validation

Offline: PYTHONPATH=tools python -m bigcherry patch-lint patches/1217_rd44_graph_opt_default_rdna35; patch-rebase-check --focal-overlay 1217_rd44_graph_opt_default_rdna35 --source bigcherry-tuning (confirm REQUIRES on 1215/1216 is declared). Hardware (on Brutus, gfx1151 only, not run here): broad dense/MoE/GDN/MTP on/off regression suite with graph-capture stress and long-context edge cases per this item's existing validation section; non-gfx1151 architecture smoke control.

Offline: PYTHONPATH=tools python -m bigcherry patch-lint patches/1217_rd44_graph_opt_default_rdna35; patch-rebase-check --focal-overlay 1217_rd44_graph_opt_default_rdna35 --source bigcherry-tuning (confirm REQUIRES on 1215/1216 is declared). Hardware: gfx1151-only broad regression suite MUST run on a real gfx1151 host (not Brutus -- Brutus has no gfx1151 GPU); gfx1100/gfx1201/gfx1030 no-op smoke control runs on Brutus.

## Effort & Risk



## Standards

Last-in-chain policy change; architecture-scoped; broad evidence; no single-model extrapolation.

## Acceptance Criteria

Default-on is allowed only after hardware-wide gfx1151 confidence with no capture/output regression; otherwise remain opt-in.

## Notes

Supersedes: RD44
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd44

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH (1217_rd44_graph_opt_default_rdna35, state=untested), hard-blocked on PRBE35's qualification per this item's own acceptance criteria. Verified GGML_CUDA_CC_IS_RDNA3_5 is a real upstream macro at b11126 common.cuh:91. Noted that PRBE34/35's gathered gfx1100/gfx1201/gfx1030 evidence does not cover gfx1151/RDNA3.5, the actual target architecture here -- dedicated gfx1151 evidence is still required and not yet gathered. No GPT design session needed (mechanism already implemented, remaining work is evidence + sequencing); gateway was unavailable this session regardless (see PRBE32 notes).

2026-09-24 GPT review req_c18183e0a9034c94 applied: NOT-READY fixes applied -- added experiment-contract binding + validation.toml/producer requirement; corrected hardware plan (gfx1151 qualification cannot run on Brutus, which has no gfx1151 GPU; Brutus is gfx1100/gfx1201 fallback-control only).

## Change Log

- 2026-09-09T10:55:57.373909+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:10.094682+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.290852+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.044114+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:59:46.152511+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_030005_amd-streamfus-successors-prbe_8761
- 2026-09-10T03:00:05.767790+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:30:27.431514+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation
- 2026-09-24T02:30:53.361539+00:00 (updated-by): Updated: section:notes
- chg_20260924_023553_re-scoped-11-rdna-boost-planni_1625
- 2026-09-24T02:36:15.008920+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:39:17.918395+00:00 (updated-by): Updated: section:steps, section:files, section:validation
- 2026-09-24T04:39:56.292144+00:00 (updated-by): Updated: section:notes
