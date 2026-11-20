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

# IQ3-class Radiance expert weights — numerical, residency and critical-path feasibility before implementation

## Description

Assess whether packing only routed experts at approximately three bits rather than the current Radiance four-bit representation improves effective GPU residency and avoids enough host transfer to outweigh unpack/compute cost while meeting quality gates on gfx1100/gfx1201. GGUF IQ3 and Strata formats are not drop-in Radiance formats. Historical container sizes and KLD figures are dated and must be re-established for the exact current source/ABI and model. This item is feasibility only; converter/native kernel implementation requires a separate follow-on decision.

## Steps

1. Inventory exact current Radiance source/recipe/container loader under the active RUM ABI. Record supported formats, model/weight identity, expert/trunk bytes, pack/unpack, activation dtype and gfx1100/gfx1201 coverage.
2. Model capacity and traffic using accepted RHP04/RHP09 costs: per-device reserves, newly resident expert pairs, route probabilities, avoided host bytes by phase, KV/head budget, layout and uncertainty.
3. Compare a pinned expert-only three-bit candidate against current four-bit and common high-precision references with trunk/tokenizer/KV/MTP held fixed. Predeclare quality bounds; report top-1, TV/KLD tails and representative coding-task quality, not mean KLD alone.
4. Evaluate candidate load/format and dequant/GEMV/GEMM shape with existing replay/fixtures. Record row shapes, rounding, accumulation, bytes, instructions and unsupported paths; prohibit silent aliases or fallback.
5. Compute a conservative end-to-end benefit bound from measured ENG01/RHP09 exposed critical path, subtracting unpack/compute/wait costs. Reject if no credible benefit exceeds paired noise.
6. Record GO/NO-GO. Only after positive quality, residency and kernel evidence propose a separate managed implementation slice with current-model, long-context, MTP and paired serving gates.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline round-trip/reference and adversarial row/ID checks bind outputs to pinned inputs. Capacity/traffic simulation covers relevant device-memory pressure and reports uncertainty. Hardware evidence, if later authorized under a GPU lease, includes current model/ABI, KLD tails, MTP acceptance and ABBA serving metrics; offline results alone do not claim activation or E2E speed.

## Effort & Risk



## Standards



## Acceptance Criteria

A pinned current-build model/source has quantified expert size, residency and traffic benefit with uncertainty and a predeclared minimum useful gain. Expert-only quantization meets an explicit quality budget against a common high-precision reference and representative coding-task check. Current loader/kernel compatibility for gfx1100/gfx1201 is demonstrated or a precise no-go is recorded. Exposed critical-path benefit is justified from ENG01/RHP09. No converter/native kernel implementation is implied by feasibility.

## Notes

References: Paiton W3A8 (all-resident on 2 x 32 GiB, quality claims given for their 4-bit tune only), Strata IQ3_XXS (GGUF) on RDNA2. Strata's IQ/K-quant kernels cannot read radiance's container directly. Related: RCS07, RHP07, RAD10.

## Change Log

- 2026-10-18T14:43:51.861513+00:00 (created-by): Created by agent

## Reviews

- RV4237
- 2026-11-20T02:21:53.155223+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:validation, section:acceptance_criteria
