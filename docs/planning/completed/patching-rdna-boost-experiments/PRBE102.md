---
id: PRBE102
order: 0
plan: patching-rdna-boost-experiments
state: completed
created-at: '2026-09-12T14:46:44.752775+00:00'
breadth: ''
skill: ''
created-by: agent
---

# Register an SSM/Mamba/GDN-family model fixture (blocks RD13/RD50 activation checks)

## Description

RD13 (patch 1206, mul_mat+RESHAPE+add fusion) and RD50 (patch 1221, GDN chunked recurrence) both target graph shapes specific to SSM/Mamba/GDN-family models. Real hardware investigation on 2026-09-13 (RD13's own README) confirmed config/models.toml has zero registered models of this family -- every tier is qwen3.x/gpt-oss/ministral, all dense or dense-MoE transformers. Ran RD13's own activation-marker probe against the only non-dense-transformer model available (tierM-qwen35b-a3b-moe-mtp, a dense-attention MoE, not SSM) in both decode and prefill shapes: 0 of 0 real marker hits in either -- consistent with RD13's own authoring note that the RESHAPE-mediated add pattern requires an SSM/Mamba-family model to appear at all. This is a real missing-fixture gap, not a methodology failure.

## Steps

1. Identify a real, licensable SSM/Mamba/GDN-family GGUF model that fits available VRAM (24GB single-GPU preferred, matching RD13/RD50's existing test topology).
2. Register it in config/models.toml with a new tier id.
3. Re-run RD13's activation-marker probe (BIGCHERRY_PATCH_HIT patch=1206_rd13) against it -- decode and prefill shapes.
4. Re-run RD50's own activation/correctness evidence against the same model.
5. Persist real evidence artifacts to both patches' evidence/ directories.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

RD13's real PPL-equality correctness evidence (2026-09-13, current pin b10901) already PASSES independently of this gap -- this item blocks only the activation-marker leg for both patches, not their correctness evidence.

**CLOSED (2026-09-13, resolved without new model registration).** GPT identified that `tierA-qwen4b-q6k` (already registered in config/models.toml, mislabeled 'dense tier') is actually a dense+GDN hybrid -- Qwen3.5-4B's real architecture has 24 of 32 layers as Gated DeltaNet recurrent layers. Reran RD13's activation probe against this existing model: subject_hit=1, control_hit=0, a clean real positive/negative split. RD13's activation leg is resolved with no new registration required. Corrected config/models.toml's inaccurate 'dense tier' note for tierA-qwen4b-q6k. RD50 (1221) remains separately blocked (its chunked-recurrence kernel is explicitly gfx1151-only, which this project doesn't have hardware for -- unrelated to model availability).

## Change Log

- 2026-09-12T14:46:44.752775+00:00 (created-by): Created by agent

## Ledger-events


- chg_20260912_144740_refreshed-patch-1206-rd13s_7649
- 2026-09-12T14:47:40.653615+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T20:05:05.159621+00:00 (updated-by): Updated: section:notes
- 2026-09-12T20:05:07.332956+00:00 (state-transition): State: pending → completed
- chg_20260912_200539_closed-the-missing-ssm-model-g_5054
- 2026-09-12T20:05:39.199453+00:00 (updated-by): Updated: section:ledger-events
