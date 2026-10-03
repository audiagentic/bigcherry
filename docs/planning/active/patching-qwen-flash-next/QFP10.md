---
id: QFP10
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:22:48.527804+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: M
---

# 1273 IQ MMVQ tuning extended to Flash-Next expert types (IQ3_S/IQ4_NL) incl. the MoE multi-token path

## Description

1273_iq_mmvq_rdna_tuning (reconciled 2026-10-03 onto 0600 + 1241; requires both; 1274 anchor adjusted to coexist) tunes VDR/nwarps for IQ4_XS/IQ3_XXS only. GPT delivered IQ3_S vdr1 and IQ4_NL vdr1 vec-dot variants with index-coverage proofs (req_407fe967d4b44af9). Blocker: MUL_MAT_ID with ncols_dst > 1 goes through mul_mat_vec_q_moe_launch -> mul_mat_vec_q_moe, which hardcodes get_vdr_mmvq/get_vec_dot_q_cuda, so MTP verify (always ncols 4) expert launches never see 1273's variants; only single-token expert decode does.

## Steps

1. Confirm the GGUF's routed-expert tensor types (gguf dump) - do not assume IQ3_S/IQ4_NL. 2. Add the GPT vdr1 variants + selector/table entries. 3. Thread iq_vdr through mul_mat_vec_q_moe<> and its launch. 4. Microbench actual expert shapes on gfx1100 and gfx1201 separately; gate on >=10% MMVQ gain on the limiting rank (R9700).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Microbench per arch; ms/step ABBA on profile v2; greedy/KLD unchanged.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

RV4214 rank #2 (GPT est. +1-3%). Online: upstream RDNA3 MMQ tuning helped IQ4_NL prefill ~9-10% but not tg; AMD mmvdq (no q8_1 activation) is the decode idea worth porting to IQ types.

## Change Log

- 2026-10-03T15:22:48.527804+00:00 (created-by): Created by agent
