---
id: PRBE41
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:18.783942+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# AMD-SSM-001: Channels-major SSM_CONV input mode

## Description

TODO. GPT unavailable this batch (see PRBE39/40 notes on req_6a45d0917b114112/req_3710dbae73fb49be) -- design authored directly from verified b11126 source. Relevance: no existing patch (grep for RD49 in patches/*/patch.toml = no hits). Not upstream-absorbed: ggml_ssm_conv's public signature at b11126 (ggml/include/ggml.h:2524, confirmed unchanged) still takes only (ctx, sx, c) with no layout parameter, and the CUDA kernel ssm_conv_f32 (ggml/src/ggml-cuda/ssm-conv.cu:6-56, read in full) hardcodes time-contiguous-within-channel addressing (`x_block[tid*stride_x + i]`, tid selects channel, i walks time within a channel's row) -- i.e. today's only supported layout is what the item calls time-major. Disposition: TODO, large (L) scope confirmed by direct inspection -- touches a public ggml API, CPU reference, and both short/long-token CUDA/HIP kernel variants.

## Steps

1. Add a layout parameter to ggml_ssm_conv's signature (ggml/include/ggml.h + ggml/src/ggml.c) -- e.g. `enum ggml_ssm_conv_layout { GGML_SSM_CONV_LAYOUT_TIME_MAJOR = 0, GGML_SSM_CONV_LAYOUT_CHANNELS_MAJOR = 1 }` stored in op_params (ggml ops pass small scalars via op_params int array, see e.g. ggml_rope's op_params usage for precedent), defaulting to TIME_MAJOR so every existing caller is unaffected without touching call sites. 2. Add a CPU reference implementation for the channels-major layout in ggml_compute_forward_ssm_conv (ggml/src/ggml-cpu/ops.cpp or equivalent) mirroring the existing time-major loop but swapping which dimension is inner-contiguous. 3. Add a channels-major CUDA/HIP kernel variant in ssm-conv.cu (new template parameter or a runtime branch reading a new op_params layout field), for both the short-token (ssm_conv_f32) and long-token (ssm_conv_long_token_f32) paths -- both currently exist at ssm-conv.cu:6 and :61 and both hardcode the same addressing scheme, so both need the alternate-layout branch. 4. Add a per-backend support predicate (a backend that does not implement channels-major must report GGML_STATUS_FAILED or fall back to a CPU copy -- follow the existing ggml backend capability-query convention, e.g. supports_op). 5. Switch ONLY the specific DeltaNet/Qwen hybrid graph-construction call sites that benefit (do not touch other ggml_ssm_conv callers) to request GGML_SSM_CONV_LAYOUT_CHANNELS_MAJOR, removing their preceding transpose/cont. 6. Add direct backend-op tests (test-backend-ops.cpp test_ssm_conv, already exists for the time-major case -- extend its parameterization with a layout axis) covering d_conv 3/4/9, both layouts, short/long tokens, single/multi sequence. 7. Add model-level determinism/PPL tests for the switched callers (qwen35 dense, qwen3next) at both old and new layout to confirm bit-identical old-layout behavior and correct new-layout behavior.

## Detailed Solution & Technical Design

Backward compatibility is achieved by making the new layout parameter default to the existing (0 = time-major) behavior everywhere op_params is zero-initialized, so every caller that does not explicitly request channels-major is bit-identical to today. The CUDA kernel change is localized: ssm_conv_f32's addressing `x_block[tid*stride_x + i + d_conv - 1]` (tid = channel, walks time contiguously within a channel row, i.e. nb0 is the time-stride) becomes, for channels-major, an addressing scheme where nb0 is the channel-stride and the per-block iteration instead walks channels contiguously -- structurally this means swapping which of src0_nb0/src0_nb1 the kernel treats as the fast-moving index, which the existing template's split_d_inner/d_conv structure can accommodate via a compile-time or runtime layout switch without restructuring the outer block/thread grid (block.x = sequence, block.y = channel-chunk, thread = channel-within-chunk stays the same either way; only the inner x[j]/w[j] gather addressing changes). The CPU reference must be added first as the correctness oracle before the CUDA path, per this project's standard op-implementation order.

## Code Samples & Guidance

ggml/include/ggml.h (op_params-based layout flag, following existing small-scalar-via-op_params precedent used elsewhere in this header):\n```c\nGGML_API struct ggml_tensor * ggml_ssm_conv(\n        struct ggml_context * ctx,\n        struct ggml_tensor  * sx,\n        struct ggml_tensor  * c,\n        int                   layout /* 0 = time-major (default/existing), 1 = channels-major */);\n```\nNote: changing the signature itself would require migrating every call site immediately per this project's no-legacy-shim standard (docs/planning CLAUDE.md doctrine) -- so ALL current callers of ggml_ssm_conv (grep `ggml_ssm_conv(` across src/models/*.cpp at implementation time) must be updated to pass layout=0 explicitly in the same change, not left to an implicit default parameter.\nggml/src/ggml.c ggml_ssm_conv(): store `layout` into `result->op_params[0]` via `ggml_set_op_params_i32(result, 0, layout);` (existing op_params helper pattern).\nggml/src/ggml-cuda/ssm-conv.cu: real base kernel signature and addressing to branch on (verified at b11126, lines 6-19):\n```cpp\ntemplate <bool apply_silu, size_t split_d_inner, size_t d_conv>\nstatic __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr, const float * bias_ptr,\n    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,\n    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2, const int64_t n_t) {\n    ...\n    const int stride_x = src0_nb1 / sizeof(float);  // time-major: nb1 is the channel-stride, walk is contiguous in nb0 (time)\n    ...\n}\n```\nFor channels-major, add a parallel kernel (or `if constexpr` branch keyed off a new template bool `channels_major`) that reads `stride_x = src0_nb0 / sizeof(float)` style addressing instead (time becomes the outer/nb1-strided dimension, channel becomes contiguous) -- exact index algebra must be worked out against the real tensor shape ggml_ssm_conv will produce for channels-major sx (ne0=d_inner, ne1=n_t+d_conv-1, ne2=n_s) before finalizing, this is a genuine kernel-design task for whoever implements it, not a mechanical anchor copy.

## Files

ggml/include/ggml.h (ggml_ssm_conv signature), ggml/src/ggml.c (ggml_ssm_conv, op_params encoding), ggml/src/ggml-cpu/ops.cpp or ops.cpp-equivalent (CPU reference), ggml/src/ggml-cuda/ssm-conv.cu + ssm-conv.cuh (CUDA/HIP kernels, both short and long-token variants), backend supports_op predicates, DeltaNet/Qwen hybrid graph-construction call sites in src/models/*.cpp (exact files TBD by grep at implementation time), tests/test-backend-ops.cpp (test_ssm_conv layout parameterization), new package patches/<order>_rd49_ssm_conv_channels_major/.

## Validation

Offline: patch-lint, patch-rebase-check. Direct op: both layouts x d_conv 3/4/9 x representative d_inner x short/long tokens x single/multi sequence, CPU and HIP, exact numeric parity between the two layouts' logical results (same math, different memory order) and bit-identical time-major output vs pre-patch baseline. Model: qwen35 dense (time-major, must be unaffected), qwen3next/hybrid callers switched to channels-major (deterministic/PPL parity vs a reference run). Composition: standalone with PRBE42 absent; do not assume PRBE42 exists yet. Hardware (Brutus, not run here): ubatch-sensitive prefill timing 256..4096 on the switched callers, decode must stay neutral (no regression) since decode path is unaffected by this API-level change.

## Effort & Risk

L (confirmed, matches the item's own original estimate) -- shared public API touched by every ggml_ssm_conv caller (signature change requires updating all callers per no-legacy-shim standard), two full kernel variants (short/long token) need the new addressing path, and CPU-reference-first discipline adds sequencing risk if skipped.

## Standards

No legacy/backward-compat shims: the signature change updates every caller in the same commit (explicit layout=0 args), not an optional/defaulted parameter. CPU reference required before HIP. Backend fallback for non-implementing backends must fail closed, not silently produce wrong output.

## Acceptance Criteria

Acceptance requires backward-compatible channels-major and time-major SSM_CONV behavior, 90/90-class direct parity, CPU+HIP correctness with clean fallback, real multi-device/meta correctness, correct composition with PRBE42, and stable prefill evidence with decode neutral.

## Notes

Supersedes: RD49
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd49

Supersedes RD49. PRBE21 is a distinct SSM conv_input concat optimization; cross-reference but do not merge.

2026-09-24 relevance at b11126: no existing patch (RD49 grep = no hits). Not upstream-absorbed (ggml_ssm_conv signature at ggml/include/ggml.h:2524 and CUDA kernel addressing at ggml/src/ggml-cuda/ssm-conv.cu:6-56 both read directly and confirmed time-major-only, no layout parameter exists). GPT design requests for this batch (req_6a45d0917b114112, req_3710dbae73fb49be) were unusable (failed/unresponsive) -- this plan was authored directly by the planning agent from verified source. PRBE21 (SSM conv_input concat) is a distinct, separate optimization per the item's own notes -- do not merge.

## Change Log

- 2026-09-09T10:56:18.783942+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:34.146919+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.314431+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.077374+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:04:24.733208+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- chg_20260910_030451_carried-forward-the-detailed-s_2071
- 2026-09-10T03:04:51.318883+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:05:49.720099+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030619_removed-migration-placeholder_7703
- 2026-09-10T03:06:19.327420+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:45:40.709189+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
