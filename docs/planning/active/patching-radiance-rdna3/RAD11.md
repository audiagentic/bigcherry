---
id: RAD11
order: 11
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-18T14:43:51.861513+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Lower-bit (IQ3-class) expert quantisation for radiance containers: feasibility and quality

## Description

Owner question 2026-10-19: would IQ-style quantisation, as in GGUF, help radiance models? The Flash-Next container is already a 4-bit expert format (fp8 trunk, iq4r experts, 114 GiB). A 3-bit-class expert format (IQ3_XXS/IQ3_S style codebooks, or Paiton's W3A8) would shrink the experts by roughly a quarter, so more of them stay GPU-resident on our 80 GiB of cards (R9700 32 + 2 x XTX 24) and fewer cross the x8 host link. Residency is our main lever (smaller expert pool gave +15% long-prompt decode; host-resident experts bound prefill). Costs: codebook/lookup dequantisation in the kernels (per arch: gfx1100, gfx1201, future gfx1030), calibration/importance data, a converter in radiance's quantiser tooling, and a quality budget (KLD harness exists).

## Steps

1. Inventory: which formats radiance's quantisers and kernels already support (iq4r, mxfp4, fp8 block) and what the container loader accepts; read the converter source and the 1.8.0 release notes.
2. Offline quality study: quantise experts only (trunk unchanged) to 3-bit variants from the same source weights and score with the KLD harness (helper-kld.json style, stored reference): kld_mean, p99, top-1, against the current 4-bit baseline (0.0633 / 0.673 / 88.3%).
3. Kernel feasibility in the replay harness: dequant + grouped GEMM microbench per arch with row sweeps, parity vs CPU reference, bytes moved.
4. Capacity arithmetic: resident pairs per tier and host-resident share under 3-bit versus 4-bit; predicted decode/prefill effect from the heat model (RHP07).
5. Decide: build the converter and kernels only if quality and arithmetic justify it; otherwise record as evaluated.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

KLD within an agreed budget of the 4-bit baseline; replay microbench shows the 3-bit expert GEMM no slower per byte than 4-bit; the capacity model shows a material cut in host-resident experts.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

References: Paiton W3A8 (all-resident on 2 x 32 GiB, quality claims given for their 4-bit tune only), Strata IQ3_XXS (GGUF) on RDNA2. Strata's IQ/K-quant kernels cannot read radiance's container directly. Related: RCS07, RHP07, RAD10.

## Change Log

- 2026-10-18T14:43:51.861513+00:00 (created-by): Created by agent
