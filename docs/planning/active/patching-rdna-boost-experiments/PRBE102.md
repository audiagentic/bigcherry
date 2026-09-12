---
id: PRBE102
order: 0
plan: patching-rdna-boost-experiments
state: pending
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

## Change Log

- 2026-09-12T14:46:44.752775+00:00 (created-by): Created by agent
