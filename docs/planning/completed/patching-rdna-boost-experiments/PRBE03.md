---
id: PRBE03
order: 0
plan: patching-rdna-boost-experiments
state: deprecated
created-at: '2026-09-09T10:53:39.131985+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Evaluate RDNA4 expanded WMMA flash-attention configurations

## Description

OBSOLETE (deprecated). PRBE03 qualified the gfx1201-only expanded WMMA flash-attention config (RD06), which was patch 1203's RD06 slice. Real Brutus hardware measurement (PA39, retry-1, pin 28ff0958/b10901, gfx1201 device 2) already ran this exact contract: activation and backend-reference correctness PASS, but performance FAILED its own threshold -- target_kernel_gain_pct point -0.0154%, CI95 low -0.0745%, required CI95 low >= 0.5%. The head-576 WMMA expansion gives no measurable gain. 1203 (the bundle) is now `rejected`; PA41 explicitly records the RD06 remediation as deferred, not resumed now.

## Steps

- Resolve baseline plus PRBE02, then apply only the expanded RDNA4 configuration hunk from patch 1203.
- Run per-shape and softcap cases on gfx1201 with balanced repeated treatment/control measurements.
- Inspect resource use and occupancy and compare outputs to the correctness-qualified baseline.
- Run explicit gfx1100 non-selection/non-regression controls and retain architecture rejection evidence.
- Record resolved identity, environment, shape matrix, repetitions and promotion decision.

## Detailed Solution & Technical Design

Use a campaign arm whose sole intended difference is the expanded RDNA4 configuration. Keep native BF16 and the PRBE02 correctness prerequisite explicit in the resolved identity; no cross-architecture default is permitted.

## Code Samples & Guidance



## Files

patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq; gfx1201 WMMA configuration; per-shape/softcap campaign artifacts; occupancy reports; gfx1100 control evidence.

## Validation

None required -- item is closed on existing real evidence, no further hardware runs needed unless a redesigned RDNA4 WMMA config is proposed later under a new identity.

## Effort & Risk



## Standards

Fail closed on performance threshold; no re-litigation without a new, materially different design.

## Acceptance Criteria

Only the correctness-qualified gfx1201 arm is evaluated; every selected shape passes output checks and the registered performance/evidence gate; gfx1100 cannot select it and shows no regression; otherwise reject/quarantine the configuration.

## Notes

Supersedes: RD06
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd06

2026-09-11: started a real gfx1201 build+bench campaign for patch 1203 (see PRBE02's notes for the same in-flight run -- same build, both items' evidence come from it). CAVEAT: this is a single-shape decode/PPL pass via llama-bench/llama-perplexity, not yet the per-shape/softcap matrix with balanced repeats and explicit gfx1100 non-selection controls this item's own acceptance criteria require. First real signal only, not closure.

REAL RESULT 2026-09-12: gfx1201 PPL-equality (see PRBE02's notes, same run): PASS, sigma=1.35, well under 3.0 threshold. Performance (llama-bench, pp512/pp2048/tg128, -fa on, r=5) launched, results pending -- see this item's own performance acceptance criteria (per-shape/softcap matrix with balanced repeats and gfx1100 non-selection control) once numbers land. Correctness confirmed; performance claim still being measured.

REAL PERFORMANCE RESULT 2026-09-12 (llama-bench, gfx1201, tierA-qwen4b-q6k, -fa on -p 512,2048 -n 128 -r 5): pp512 +3.4% (5260.12->5439.31 t/s), pp2048 +6.2% (5102.77->5417.97 t/s, very tight stddev on both arms making this a credible signal), tg128 +0.0% (92.63 t/s both, correctly unaffected -- decode doesn't touch flash-attn prefill/Q6_K mmq). Real, positive first signal for RD06's config-enablement claim. Still missing per this item's own acceptance criteria: per-shape/softcap matrix, balanced statistical repeats beyond r=5, and an explicit gfx1100 non-selection/non-regression control proving the expanded config doesn't activate there. See patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq/README.md for full writeup.

append

2026-09-24 relevance at b11126: DEPRECATED. RD06's expanded gfx1201 WMMA FA config (patch 1203 slice) was conclusively measured on real gfx1201 hardware (PA39): CI95 low -0.0745% vs required >=0.5% -- no gain. 1203 is `rejected`; PA41 defers RD06 remediation rather than reopening it now. PRBE03 as originally scoped (qualify this exact config) is closed; do not duplicate PRBE110 (which only re-extracts the passing RD05/RD07 slices, explicitly NOT RD06). No GPT design request needed -- disposition follows directly from existing real hardware evidence already recorded in this item's own notes.

## Change Log

- 2026-09-09T10:53:39.131985+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:10:18.220021+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.138725+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.811061+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:30:18.912456+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023037_three-rdna-boost-successors-no_6965
- 2026-09-10T02:30:37.084094+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T23:50:30.925613+00:00 (updated-by): Updated: section:notes
- chg_20260911_235103_caught-myself-running-a-real-h_5912
- 2026-09-11T23:51:03.373803+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T03:20:09.369112+00:00 (updated-by): Updated: section:notes
- 2026-09-12T03:21:06.351281+00:00 (updated-by): Updated: section:notes
- 2026-09-24T02:26:41.916056+00:00 (updated-by): Updated: section:description, section:validation, section:standards, section:notes
- 2026-09-24T02:26:54.476380+00:00 (updated-by): Updated: section:notes
- 2026-09-24T02:26:59.903653+00:00 (state-transition): State: pending → deprecated
