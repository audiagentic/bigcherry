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

TODO. GPT unavailable this batch (see PRBE39 notes) -- design authored directly from verified b11126 source. Relevance: no existing patch (grep for RD56 = no hits). Not upstream-absorbed: b11126's ggml/src/ggml-cuda/common.cuh warp-reduction helpers (warp_reduce_sum(int/float/float2/half2), read directly at common.cuh:459-495) are all implemented via `__shfl_xor_sync` (HIP's portable shuffle-xor intrinsic), with no RDNA DPP (`__builtin_amdgcn_mov_dpp` / `__builtin_amdgcn_update_dpp` / v_dpp instructions) path anywhere in ggml-cuda. Disposition: TODO, scoped to the kernels the item already names (Q6/Q8 decode hot paths).

## Steps

1. Identify the specific Q6/Q8 decode kernels that call warp_reduce_sum in their hot loop -- grep mmvq.cu/mmq-vec-dot.cuh for warp_reduce_sum call sites inside Q6_K/Q8_0 dequant-dot paths (do not touch warp_reduce_sum's generic definition, which is shared by every kernel in the codebase -- a blanket replacement there would violate the item's own 'no blanket substitution' requirement). 2. Write a NEW, separately-named helper (e.g. warp_reduce_sum_dpp) that expresses the SAME reduction pattern via RDNA DPP intrinsics (__builtin_amdgcn_update_dpp row-shuffle sequence for a full warp butterfly reduction), guarded by `#if defined(__HIP_PLATFORM_AMD__) && (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc))` at the call site, not by redefining warp_reduce_sum itself. 3. Wire the new helper into the identified Q6/Q8 kernel call sites ONLY, with the existing warp_reduce_sum() as the unconditional fallback path for every other kernel and every non-RDNA/CDNA architecture. 4. Verify generated ISA: build both variants with `hipcc --save-temps` (or `-S`) for the target kernel and diff the emitted v_dpp vs v_shfl instruction sequences and DS (LDS) usage. 5. Add exact-output tests (existing test-backend-ops MUL_MAT cases at the Q6_K/Q8_0 shapes already exercise these kernels; confirm they still pass bit-identical). 6. Benchmark the targeted kernels alone (isolated microbench, not just E2E) plus E2E decode, with instruction-count/DS-usage/TG-timing evidence before deciding whether to keep the DPP path per-kernel.

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
