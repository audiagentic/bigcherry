---
id: PRBE47
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:45.120291+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-HIP-002: AMD DPP/native shuffle path

## Description

TODO, corrected per GPT review (req_e17e0bf5a68c48d5). Source-location assumption was wrong: Q6_K MMVQ's vec_dot_q6_K_q8_1* in vecdotq.cuh performs LOCAL DP4A accumulation only and does not call warp_reduce_sum. The actual warp_reduce_sum call sites for the MMVQ hot path are generic, type-templated reductions in mmvq.cu (confirmed: mmvq.cu:795,798,937,940, `tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i])` etc.), SHARED by every quant type, not Q6_K/Q8_0-specific. A blanket DPP swap at these sites would affect every quant type's reduction, contradicting the item's own 'no blanket substitution' requirement. DPP reduction sequence and compile guard remain TBD pending real ISA verification.

## Steps

1. Do NOT add a new warp_reduce_sum call inside vecdotq.cuh's vec_dot_q6_K_q8_1* (it does not reduce across the warp there -- confirmed by direct read). 2. Target the real call sites instead: mmvq.cu:795/798/937/940 (and any other warp_reduce_sum<warp_size> calls in this file's MMVQ dispatch), which are generic and shared across all quant types. 3. Implement a verified wave32 DPP helper (warp_reduce_sum_dpp) as before. 4. Specialize ONLY these mmvq.cu reduction sites with `if constexpr (type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q8_0)` (or an equivalent type-gated branch keyed off the kernel's template type parameter) plus an AMD wave32 compile guard, calling warp_reduce_sum_dpp for the targeted types only; retain warp_reduce_sum<warp_size> unconditionally for every other type/architecture -- this satisfies the 'no blanket replacement' requirement since the swap is type-gated at the existing generic call site, not a redefinition of warp_reduce_sum itself. 5. Validate generated ISA (hipcc --save-temps) for both a targeted (Q6_K) and non-targeted (e.g. Q4_0) type to confirm the branch correctly isolates codegen. 6. Add a non-target quant type as an explicit negative control in test-backend-ops (must show unchanged codegen/behavior).

## Detailed Solution & Technical Design

b11126's warp_reduce_sum is a template-free set of overloads in common.cuh used pervasively (attention, MMQ, MMVQ, norm kernels, etc) via `__shfl_xor_sync`, which HIP lowers to `ds_permute`/`ds_bpermute` or `v_shfl` style cross-lane moves depending on target -- on RDNA, native DPP (data-parallel-primitives) row-shuffle instructions can sometimes execute the same butterfly pattern with fewer cycles and no LDS round-trip that HIP's shuffle emulation may need on some RDNA generations, but this is NOT guaranteed to win (RDNA3/4 shuffle lowering has improved across ROCm versions) -- hence the item's own insistence on measuring generated ISA and runtime per kernel rather than assuming a win. The safest implementation shape is a parallel, separately-named helper selected only at specific hot call sites via an architecture-guarded macro/if-constexpr, never a redefinition of the shared warp_reduce_sum (which would silently change every kernel's codegen, violating the item's 'no blanket replacement' requirement and dramatically raising regression surface).

## Code Samples & Guidance

Real b11126 anchor (ggml/src/ggml-cuda/common.cuh:472-479, exact):\n```cpp\nstatic __device__ __forceinline__ float warp_reduce_sum(float x) {\n#pragma unroll\n    for (int offset = 16; offset > 0; offset >>= 1) {\n        x += __shfl_xor_sync(0xffffffff, x, offset, width);\n    }\n    return x;\n}\n```\nNew, separately-named DPP variant (sketch -- exact DPP row-op sequence needs verification against ROCm's device intrinsics header for the target ROCm version, this is a genuine implementation task not a copy-paste):\n```cpp\n#if defined(GGML_USE_HIP) && defined(__gfx1100__) || defined(__gfx1201__)\nstatic __device__ __forceinline__ float warp_reduce_sum_dpp(float x) {\n    x += __builtin_amdgcn_update_dpp(0.0f, x, 0x111, 0xf, 0xf, false); // row_shr:1 style butterfly step, repeat log2(warpSize) times\n    ... // full butterfly sequence TBD, verify against real amdgcn DPP ISA docs\n    return x;\n}\n#endif\n```\nCall-site gating (in the identified Q6/Q8 kernel, not in common.cuh):\n```cpp\n#if defined(GGML_USE_HIP) && (defined(__gfx1100__) || defined(__gfx1201__))\n    sum = warp_reduce_sum_dpp(sum);\n#else\n    sum = warp_reduce_sum(sum);\n#endif\n```\npatch.toml: id="<order>_rd56_dpp_shuffle_targeted", state="untested", backend="hip", plan-item="PRBE47", experiment-contracts=["UP-HIP-002-DPP-TARGETED-KERNELS"].

## Files

ggml/src/ggml-cuda/common.cuh (new warp_reduce_sum_dpp helper, additive only), ggml/src/ggml-cuda/mmvq.cu and/or mmq-vec-dot.cuh (call-site swap at the identified Q6/Q8 hot loops, exact function names TBD by grep at implementation time), tests/test-backend-ops.cpp (existing MUL_MAT Q6_K/Q8_0 cases as regression oracle), new package patches/<order>_rd56_dpp_shuffle_targeted/.

## Validation

Offline: patch-lint, patch-rebase-check. Correctness: exact output parity for every targeted kernel/shape (existing test-backend-ops MUL_MAT cases at Q6_K/Q8_0 must stay bit-identical). ISA: hipcc-generated assembly diff showing v_dpp usage and instruction-count/DS-usage delta per kernel. Hardware (Brutus, not run here): targeted-kernel microbench + E2E Qwen decode on gfx1100/gfx1201, non-RDNA/generic-fallback control unaffected.

## Effort & Risk

M (unchanged) -- lane-topology correctness for DPP row ops is easy to get subtly wrong (silent wrong-sum bugs), and compiler lowering of __shfl_xor_sync already varies by ROCm version, so a measured non-win is a real possible outcome per the item's own acceptance criteria.

## Standards

No blanket intrinsic substitution -- new helper is additive and call-site-gated, never a redefinition of the shared warp_reduce_sum. Non-RDNA portability preserved via #if guards. Generated-ISA evidence required before promotion.

## Acceptance Criteria

Acceptance requires exact output parity, generated-ISA and DS-use evidence, repeatable targeted-kernel and E2E improvement, and generic fallback on unsupported architectures/patterns; no blanket replacement.

## Notes

Supersedes: RD56
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd56

Supersedes RD56.

2026-09-24 relevance at b11126: no existing patch (RD56 grep = no hits). Not upstream-absorbed -- common.cuh's warp_reduce_sum family read directly at lines 459-495, confirmed __shfl_xor_sync-only, no DPP path exists anywhere in ggml-cuda (grep for amdgcn_*dpp across ggml-cuda = zero hits). GPT design requests this batch were unusable; plan authored directly from verified source.

2026-09-24 GPT review req_e17e0bf5a68c48d5 applied: corrected the target call sites from vecdotq.cuh's vec_dot_q6_K_q8_1* (which does not call warp_reduce_sum) to the real generic, type-templated reduction sites in mmvq.cu (confirmed at mmvq.cu:795,798,937,940); specified an if-constexpr type-gated specialization at those sites rather than a blanket redefinition.

## Change Log

- 2026-09-09T10:56:45.120291+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:58.680914+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.341721+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.115748+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:08:59.693148+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-10T03:09:19.555246+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030930_repaired-two-more-active-patch_7368
- 2026-09-10T03:09:30.744289+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:46:57.296925+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-24T05:08:29.772808+00:00 (updated-by): Updated: section:description, section:steps, section:notes
