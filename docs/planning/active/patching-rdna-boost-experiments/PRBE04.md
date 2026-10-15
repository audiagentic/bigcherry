---
id: PRBE04
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:53:43.516065+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---
# RD07 Q6_K MMQ — downstream safety and promotion gate

## Current disposition (2026-10-14, b11552)

**Pending, blocked on PRBE110/1267 source and opportunity admission.** The prior plan's "PRBE110 must deliver a new package" is superseded: `engines/llamacpp/patches/1267_rd07_q6k_mmq_scale_fold/` exists, was measured at b11126, and is `untested` plus `known_broken` on b11552. Do not create a second RD07 package or independently re-anchor 1267 here. PRBE110 is the sole implementation owner; this item is the downstream safety/qualification decision.

**12-hour exclusion:** last substantive independent PRBE04 edit 2026-10-08 06:47 UTC, PRBE110 same date, 1267 package/evidence relocation 2026-10-09 04:20 UTC. No independent Q6_K scale-fold patch, benchmark, PR or queued experiment was identified in the preceding 12 hours. Active Radiance/RAD02/RAD03/RAD04 implementation and recent BCOP190–210 paths remain untouched.

## Proven blockers and evidence

- b11552 `ggml/src/ggml-cuda/mmq.cu` dispatches `mul_mat_q_case<GGML_TYPE_Q6_K, GGML_PREC_Q8>(ctx, args, stream)`; 1267's marker expects the removed `<GGML_TYPE_Q6_K>(..., forced_J)` call. Its `mmq.cuh` `J_max` anchor also vanished. The current 0300 framework selects J **before** activation padding, so the old 1267 J-max knob must not be independently resurrected. `dispositions/1267_rd07_q6k_mmq_scale_fold.json` already records `known_broken`.
- b11126 archived gfx1201 pp512 +2.4675%, CI95 [2.2776%, 2.6508%]; tg128 -0.0331%, CI95 [-0.0592%, -0.0071%]. Gfx1100/gfx1030 failed to establish benefit. The gfx1201 record's contract pass coexists with `validation_disposition=incomplete`, `eligible_for_validated_state=false` and failed performance/control check fields. This is **historical, pin-bound, not validated or production gain**.
- The proposed folded F32 expression is not algebraically bit-identical in general. A 200,000-case host model yielded 38,735 differing F32 bit patterns; no GPU mismatch was reproduced. Current Flash-Next IQ4_XS/Q8_0 does not execute Q6_K MMQ, so first prove a relevant Q6_K model/lane and exposed >=3% E2E ceiling.

## One downstream qualification sequence

1. **Admission from PRBE110:** require a proven executed Q6_K MMQ route and measured critical-path ceiling; terminalize `NO_RELEVANT_Q6K` or `NO_EXPOSED_GAIN` without hardware if absent.
2. **Composition from PRBE110:** require current-pin 1267 apply/lint, existing 0300 preallocation-J owner, one subject-only activation marker, no stale `GGML_CUDA_MMQ_J_MAX` patch surface, qualified gfx1201 gate, and stock fallback on gfx1100/gfx1030. No new tuning selector.
3. **PEF01/HI71 safety:** preserve the exact gfx1100 Q6_K J=112/fallback=0 quarantine; test eligible/illegal J and padded-Y tail bounds at each physical GPU. A clean generic MMQ test is **not** evidence that the quarantined signature is safe. Check graph on/off, repeated requests, multi-ubatch, VRAM/scratch, backend Q6_K ops and negative Q8_0/F16 controls.
4. **Numerical:** same-process target logits/KLD and full-vocabulary greedy checks against the unchanged Q6_K control. Report max ULP and distribution of deltas for scale reassociation; do not demand impossible blanket bit identity without checking actual GPU arithmetic. Missing work or silent fallback is failure.
5. **Performance:** only after safety and activation pass, consume PRBE110's one four-session gfx1201 campaign (>=10 paired rounds/session) with pp512/pp2048 and tg128/other-quant controls. Promotion: pp512 CI95-low >=2%, controls <=1% regression, and representative E2E >=3% for production. No repeated gfx1100/gfx1030 gain campaigns.

**Ownership:** PRBE110/1267 source and architecture gate; PRBE04 downstream PEF01/HI71 and acceptance; 0300 J selection; 1006 Q6_K codegen; existing validation producer/evidence. Retire all older PRBE04 "new patch TBD" and duplicate queue instructions. Terminal outcomes: `NO_RELEVANT_Q6K`, `NO_EXPOSED_GAIN`, `SOURCE_REBASE_BLOCKED`, `SAFETY_OR_NUMERIC_FAIL`, `NO_VALIDATED_GAIN`, `NOT_PRODUCTION_RELEVANT`, `PROMOTABLE`.
