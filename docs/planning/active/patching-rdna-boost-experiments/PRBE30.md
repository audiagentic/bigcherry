---
id: PRBE30
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:30.594236+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-GEMM-003: K-pad F16 shadow to avoid aliasing

## Description

TODO, depends on PRBE29 -- CORRECTED dependency wiring (per GPT review): fold into PRBE29 with NO `requires` (same package), since a separate package that both depends on PRBE29's package AND is anchored via additive Edits into PRBE29's SAME package file would be a self-dependency contradiction as previously worded. K-dimension padding of PRBE29's F16 shadow for shapes (e.g. down_proj) with a measured row-stride aliasing class -- same cache-set-aliasing problem family as PRBE28/RD35, but scoped specifically to the F16 shadow buffer rather than the original quantized weight.

## Steps

1. CORRECTED per GPT review: either (a) fold this item's padding parameter directly into PRBE29's own package with no separate `requires` (since it is an additive Edit on the exact same patch.py file), OR (b) if kept as a genuinely separate package, name it patches/12xx_rd37_kpad_f16_shadow/ with `requires=["<PRBE29's final package id>"]` -- never both self-anchor into PRBE29's package AND declare a `requires` on it (that is a self-dependency contradiction present in the prior wording). PRBE29 must first expose a stable shadow-allocation helper function (with explicit row-stride/leading-dimension metadata as a parameter or return field) for PRBE30 to anchor and modify -- record that helper's exact signature once PRBE29 lands.
2. Choose affected shapes (down_proj and other candidates) by measured row-stride alias class, reusing PRBE28's causal-evidence method (rocprofv3 L2 cache-set-conflict counters) applied to the shadow buffer specifically.
3. Compare padding 0 vs one cache line against non-alias-class controls.
4. Validate F16 stride/consumer layout (the dense GEMM path reading the shadow must tolerate the padded stride) and GEMM output equivalence.
5. Measure GEMM/PP delta, shadow memory overhead, and load/working-set overhead from padding.
6. Keep padding strictly conditional on measured aliasing stride -- never unconditional.

## Detailed Solution & Technical Design

This item is a small, conditional extension of PRBE29's allocator (an optional K-padding parameter on the same allocation call), not new allocator code. It should share PRBE28's aliasing-detection and causal-evidence methodology (same L2 cache-set-conflict counter approach) rather than reinventing it, applied narrowly to the shadow buffer's row stride instead of the original quantized tensor's.

## Code Samples & Guidance

No new anchors beyond PRBE29's allocator (not yet written) -- this item's Edit() would add a stride parameter to that allocator's signature once PRBE29 exists. Patch package: an ADDITIVE Edit set on patches/12xx_rd36_f16_shadow_dense/ (PRBE29's package) gated by `requires` on PRBE29's patch id, not a separate independent patch, since the item text itself says 'not an independent shadow implementation'.

## Files

Same files as PRBE29 (ggml-cuda.cu shadow allocator, once it exists); no new files beyond a conditional padding parameter and its validation script.

## Validation

Exact layout/output equivalence; aliasing vs non-aliasing stride classes; padding-boundary sweep (0 vs one cache line); GEMM/PP delta; memory/load overhead; hard dependency check that PRBE29 is materialized first.

## Effort & Risk

S (conditional extension of an existing allocator, not new allocation logic) once PRBE29 exists; cannot start before PRBE29 is implemented.

## Standards

Dependency-aware; stride/layout correctness; conditional padding; causal isolation.

## Acceptance Criteria

Padding is selected only for proven alias classes, preserves output and yields a repeatable net benefit after memory cost; otherwise retain unpadded shadow.

## Notes

Supersedes: RD37
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd37

2026-09-24 relevance at b11126: TODO, blocked on PRBE29 (not yet implemented this batch -- design only). GPT design request for PRBE29+30+31 hit a queue-saturated gateway and was not obtained in-session; plan authored directly, deliberately kept minimal since PRBE30 has almost no independent design surface beyond PRBE29's.

2026-09-24 GPT review req_2b717df095b44703 applied: corrected the self-dependency contradiction -- the prior plan both anchored PRBE30 as additive edits into PRBE29's own package AND declared `requires` on PRBE29, which is inconsistent. Now either folds into PRBE29's package with no requires, or becomes a genuinely separate package with requires=[PRBE29's id]; PRBE29 must expose a stable shadow-allocation helper with explicit stride metadata for this item to anchor against.

## Change Log

- 2026-09-09T10:55:30.594236+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:43.525844+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.261941+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.002185+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:56:58.392222+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025719_dense-gemm-successors-prbe293_6872
- 2026-09-10T02:57:19.697607+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:34:12.638864+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:46:48.845567+00:00 (updated-by): Updated: section:description, section:steps, section:notes
