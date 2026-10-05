---
id: QFP29
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-05T09:37:29.282620+00:00'
breadth: ''
skill: advanced
created-by: claude
priority: P2
work: M
---

# Assess ROCmFPX: integer-WMMA (W4A4/IU4) batched MTP verify and MTP batching for gfx1100/gfx1201

## Description

Source: https://github.com/charlie12345/ROCmFPX (main at fb08d7cdb67082ddda15a25e64dc7ec413afc41c, 2026-09-23; standalone llama.cpp fork, MIT, ~280 commits, labelled experimental; read 2026-10-05 through a page summary only, so every number below is the project's own claim). It adds AMD-oriented 2/3/4/6/8-bit GGUF weight formats (ROCmFP2/3/4/6/8) with CPU reference, HIP and Vulkan kernels, MTP batched-verify changes and K/V cache routing (TurboQuant). Primary target is Strix Halo (gfx1151); gfx1030/gfx1100/gfx1200 are listed; the author also names an R9700 as development hardware.

Not a fit as a whole: the gains come from new weight formats that need re-quantising from full precision (our models are fixed community quants: Flash-Next UD-IQ4_XS, 27B Q8_0), headline numbers compare its own formats with each other or MTP with no-MTP on other hardware, the 2/3-bit formats trade quality for speed, and it is a fork rather than a patch set.

The one candidate: its W4A4 path uses the 4-bit integer matrix instruction v_wmma_i32_16x16x16_iu4 for batched MTP verification, claimed +18.66% mean decode on Qwen3.8-27B (41.63 -> 49.40 t/s) at about 5.4% higher perplexity than exact int8 MMQ, and stated gfx1151-only. Batched MTP verify is where Flash-Next decode time goes on Brutus.

## Steps

1. Read the actual code (not the README): the IU4 verify path, what selects it, which tensors/quant types it applies to, and whether it can serve standard quants (Q8_0, IQ4_XS, Q4_K) or only ROCmFP4.
2. Hardware: which integer WMMA instructions exist on gfx1100 (RDNA3) and gfx1201 (RDNA4) - iu4 and iu8 variants - and whether the ROCm HIP compiler exposes them there; why the project gates on gfx1151.
3. Accuracy: where the 5.4% perplexity cost comes from (4-bit activations), whether an 8-bit integer variant (W8A8 / iu8) keeps exactness or near-exactness, and what gate would replace greedy identity (it cannot pass identity by design).
4. Compare its MTP batched-verify logic with ours (1321 forced-front, 1294, 1307-1313 Q8_1 paths): anything that reduces target verify cost per speculative step without changing math.
5. Decide: port a bounded kernel path as a BigCherry patch behind a flag with a quality gate, or record as not applicable. Do not port the weight formats.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Any ported path: offline mechanics tests, activation marker, ABBA on Flash-Next 24K MTP decode and 27B, plus a stated quality gate (logit/KLD or fixed-prompt check) since output will differ. KV precision rules unchanged (f16 preferred, q8_0 minimum).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Tracked in config/external-sources.toml as charlie12345-rocmfpx. GPT deep-dive requested 2026-10-05.

## Change Log

- 2026-10-05T09:37:29.282620+00:00 (created-by): Created by claude
- 2026-10-05T09:37:46.705264+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:notes
