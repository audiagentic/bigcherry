---
id: BCOP14
order: 14
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T04:43:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Qualify upstream AMD quant-kernel register and byte-permute optimizations

## Description

Qualify two still-open upstream HIP changes against BigCherry's actual gfx1100/gfx1201 targets without creating a local architecture/dispatch framework: llama.cpp #29910 reduces Q2_K MMQ register pressure by removing the `y_df[J/nwarps]` temporary, limiting the inner J-loop to `#pragma unroll 2`, and disabling unrolling of the outer K loop; #29927 replaces emulated `__byte_perm` Q1_0 unpacking with native `__builtin_amdgcn_perm` in both MMQ tile load and MMVQ/vec-dot paths.

These are independent mechanisms and must be promoted independently. #29910 has strong spill evidence on gfx1100/gfx1200 but mixed throughput on RDNA3.5, so zero spills is not itself a promotion criterion. #29927 has large gfx906 gains but no BigCherry RDNA3/RDNA4 measurements, so it remains a hardware-qualification candidate rather than an assumed win.

## Steps

1. Current pin check: b11402/BigCherry pin still does not contain #29910 or #29927; both remain open upstream.
2. Apply each upstream diff independently to the current pin; also test both together only after the individual lanes pass correctness.
3. Build gfx1100 and gfx1201 with compiler resource diagnostics. Record VGPR/SGPR, scratch/spill bytes, LDS, occupancy and generated ISA for affected kernels.
4. #29910 Q2_K: run `test-backend-ops` for `MUL_MAT` and `MUL_MAT_ID`, then sweep ub16/32/64/128/256/512 with fixed pp2048 plus representative TG/MoE lanes. Compare baseline, exact upstream unroll policy, and at most one diagnostic `unroll 1` variant; do not create runtime dispatch for compiler pragmas.
5. #29927 Q1_0: run `test-backend-ops` for Q1_0 MMVQ and MMQ paths, then benchmark pp2048 and tg128 plus routed `MUL_MAT_ID` if Q1_0 experts are available. Verify generated ISA contains native `v_perm_b32`/equivalent and that the old emulation sequence disappears.
6. Use repeated ABBA runs on the same process/build settings. Promote per architecture only for >=3% whole-lane improvement or a >=10% affected-kernel improvement that is not offset elsewhere; reject <=1% whole-lane movement unless it removes a demonstrated pathological spill cliff.
7. If architecture crossover exists, reuse the existing HIP compile/config ownership. Do not add a BCOP-owned runtime table.

## Detailed Solution & Technical Design

### #29910 Q2_K spill reduction

Upstream changes only `ggml/src/ggml-cuda/mmq-vec-dot.cuh`. The current implementation precomputes `float2 y_df[J/nwarps]`; because J is a template parameter this extends several values' live ranges through the dot-product loops. #29910 removes that array and converts the one required scale at point of use with `__half22float2(y_ds[j*MMQ_TILE_Y_K]).x/.y`. It also changes the inner J loops from full unroll to `#pragma unroll 2` and the outer `k01` loop to `#pragma unroll 1`.

Upstream measured spill counts falling from 1387 to 0 on gfx1100 and 2041 to 0 on gfx1200. However, its gfx1152 MMA throughput is roughly neutral/slightly negative at ub64-256 (-0.4% to -1.5%) while ub16/32 improves ~34%/~20%. Therefore BigCherry should not infer throughput from spill count; the expected value is primarily removal of shape-dependent spill cliffs.

No new kernel is required. The implementation candidate is the exact upstream patch. If the current compiler produces a different crossover on gfx1100 versus gfx1201, keep the upstream source unless a reproducible >=3% whole-lane regression justifies a compile-time architecture guard using the backend's existing architecture facilities.

### #29927 native Q1_0 byte permutation

Upstream changes both `ggml/src/ggml-cuda/mmq-load-tiles.cuh` and `ggml/src/ggml-cuda/vecdotq.cuh`. Under `GGML_USE_HIP`, each 16-bit Q1_0 crumb word is expanded into four byte-selector words and passed to `__builtin_amdgcn_perm(0x000001FF, 0x000001FF, selector)`. CUDA retains `__byte_perm`.

This is deliberately applied at both ownership points: MMQ tile staging and vec-dot/MMVQ. Do not optimize only one and then attribute model-level changes to the mechanism. BigCherry's pinned code already uses `__builtin_amdgcn_perm` elsewhere, so compiler/toolchain availability is established; selector semantics and Q1_0 correctness still require backend-op tests.

Upstream gfx906 reports pp2048 2512.65 -> 3570.25 t/s (+42.1%) and tg128 66.99 -> 128.96 t/s (+92.5%), with microbench gains largest at small n. Those numbers are mechanism evidence only; RDNA3/4 use different execution resources and require local measurement.

### Consolidation boundary

BCOP14 owns qualification/evidence only. Any accepted source should be carried as the upstream patch or absorbed by the normal llama.cpp pin once merged. HIP architecture/config ownership remains where existing backend tuning lives. Do not create a permanent BigCherry Q2_K or Q1_0 dispatch registry from this item.

## Code Samples & Guidance

For #29910, preserve upstream's point-of-use scale conversion and bounded unroll exactly for the first experiment. Do not combine with other MMQ tuning before attribution is established.

For #29927, validate selector expansion exhaustively on host for all 65536 possible 16-bit Q1_0 crumb patterns against the existing unpack semantics before GPU benchmarking. This is a cheap mock/reference test and catches byte-lane mistakes independently of floating-point tolerance.

Pseudo-test:

```cpp
for (uint32_t q = 0; q <= 0xffff; ++q) {
    auto ref = q1_unpack_reference((uint16_t) q);
    auto sel = q1_build_perm_selectors((uint16_t) q);
    auto got = emulate_v_perm(0x000001ff, 0x000001ff, sel);
    REQUIRE(got == ref);
}
```

Keep this as test-only/reference logic unless upstream exposes an equivalent host helper; do not add production duplication solely for the test.

## Files

- Upstream #29910: `ggml/src/ggml-cuda/mmq-vec-dot.cuh`.
- Upstream #29927: `ggml/src/ggml-cuda/mmq-load-tiles.cuh`, `ggml/src/ggml-cuda/vecdotq.cuh`.
- Existing BigCherry HIP tuning/config owner for any proven architecture crossover; BCOP14 must not own production dispatch.

## Validation

Correctness: affected `test-backend-ops` `MUL_MAT` + `MUL_MAT_ID`; exhaustive Q1_0 selector host reference; two sequential model requests; deterministic greedy output against the same baseline build.

Performance: gfx1100 and gfx1201 separately; pp2048 and TG128 minimum; ub16-512 for Q2_K; kernel/resource evidence plus whole-lane ABBA. Record clocks/power state and reject runs with thermal/frequency drift.

Promotion: >=3% whole-lane or >=10% affected-kernel improvement with no >1% whole-lane regression, all correctness gates green. A spill fix with <=1% throughput is retained only if it removes a reproducible catastrophic shape cliff relevant to BigCherry workloads.

Stop: if exact upstream is <=1% across representative lanes and no relevant spill cliff remains, classify as wait-for-upstream rather than maintaining a local patch.

## Effort & Risk

Low implementation risk because both candidates are small upstream diffs; medium qualification effort because compiler/resource behavior is architecture- and shape-dependent. Main risk is creating permanent local tuning for a change that will likely merge upstream; avoid that by keeping qualification patches disposable.

## Standards

Use existing patch-lint/build/test conventions, current pinned llama.cpp, and the project's ABBA/performance-integrity rules. Performance evidence is invalid if correctness differs or work disappears.

## Acceptance Criteria

- #29910 and #29927 are independently classified as adopt-now, wait-for-upstream, reject, or already-upstream.
- gfx1100 and gfx1201 resource/ISA plus correctness evidence accompanies performance data.
- Q1_0 selector semantics have an exhaustive reference check or equivalent proof.
- No duplicate dispatch/config registry is introduced.
- Any local qualification patch is removable once upstream merges.

## Notes

2026-10-05 deep audit: #29910 remains open and reports Q2_K spill elimination including gfx1100 1387 -> 0 and gfx1200 2041 -> 0, but RDNA3.5 MMA throughput demonstrates a shape crossover; qualification therefore gates on actual whole-lane performance, not spill count. #29927 remains draft/open and reports large gfx906 gains; its exact diff covers both MMQ and MMVQ ownership points. Current BigCherry pin has native AMD permute usage elsewhere, reducing toolchain risk but not eliminating semantic/performance qualification.

## Related

BCOP24; llama.cpp #29910, #29927; existing HIP architecture/autotune ownership.

## Change Log

- 2026-10-05T04:56:45.106215+00:00: Confirmed both candidates absent from then-current pin.
- 2026-10-05T06:14:00+00:00: Deep implementation audit; added exact source ownership, upstream measurements, mock/reference test, architecture gates, promotion/stop criteria and consolidation boundary.
