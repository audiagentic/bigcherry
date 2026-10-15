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
priority: P3
---

# AMD-SSM-001: Channels-major SSM_CONV input mode

## Description (historical; superseded by authoritative BCOP234 gate)

TODO, corrected per GPT review (req_e17e0bf5a68c48d5). Channels-major shape/index algebra was left underspecified: ggml_ssm_conv() derives n_t=sx->ne[0]-d_conv+1 and d_inner=sx->ne[1] for the existing time-major layout -- this is WRONG for the proposed channels-major [d_inner,time,n_seq] input and must be spelled out exactly, not left for implementation-time discovery. Also: this project's no-legacy-shim doctrine means a REQUIRED new argument cannot be silently defaulted for existing callers -- the plan's prior 'defaults to TIME_MAJOR so existing callers are unaffected' framing contradicts CLAUDE.md's own no-legacy/no-backward-compat rule (every caller must be migrated explicitly, not left on an implicit default). Additionally, switching a generic model graph to channels-major requires every backend reachable by that graph to understand the new layout, not just CPU/HIP.

## Steps (historical sketch)

1. Preserve `ggml_ssm_conv(ctx, sx, c)` UNCHANGED as the time-major op (ggml/include/ggml.h:2524) -- do not touch its signature or existing callers, per the item's own precedent of keeping unrelated call sites stable. 2. Add a SEPARATE, new API `ggml_ssm_conv_ext(ctx, sx, c, layout)` (not a defaulted parameter on the existing function -- an explicit new function name, satisfying no-legacy-shim doctrine by requiring every new caller to state its layout explicitly rather than inheriting a default). 3. Define channels-major construction EXACTLY: input sx shape [d_inner, n_t+d_conv-1, n_s] (contiguous in d_inner first) -> d_inner=sx->ne[0], n_t=sx->ne[1]-d_conv+1, result tensor {d_inner, n_t, n_s}. 4. Specify exact CPU reference indices for channels-major: for output element (ic, it, is), sum over k=0..d_conv-1 of sx[ic, it+k, is] * c[ic, k] (channel-contiguous inner loop, time is the outer/strided dimension -- the mirror image of the existing time-major addressing where channel is the strided dimension). 5. Specify exact CUDA/HIP short-token and long-token kernel indices analogously (swap which of nb0/nb1 is the fast-moving index, per the existing addressing pattern in ssm-conv.cu). 6. Add a per-backend supports_op predicate: any backend not implementing GGML_SSM_CONV_LAYOUT_CHANNELS_MAJOR must report GGML_STATUS_FAILED, not silently produce wrong output. 7. EITHER implement all backends reachable by the specific DeltaNet/Qwen hybrid graphs that would call ggml_ssm_conv_ext, OR add a backend-capability-aware graph-selection mechanism that refuses to build a channels-major graph on a backend lacking support, before switching any caller. 8. Switch only the specific hybrid graph call sites that benefit, using the new ggml_ssm_conv_ext explicitly.

## Detailed Solution & Technical Design (historical sketch)

Backward compatibility is achieved by making the new layout parameter default to the existing (0 = time-major) behavior everywhere op_params is zero-initialized, so every caller that does not explicitly request channels-major is bit-identical to today. The CUDA kernel change is localized: ssm_conv_f32's addressing `x_block[tid*stride_x + i + d_conv - 1]` (tid = channel, walks time contiguously within a channel row, i.e. nb0 is the time-stride) becomes, for channels-major, an addressing scheme where nb0 is the channel-stride and the per-block iteration instead walks channels contiguously -- structurally this means swapping which of src0_nb0/src0_nb1 the kernel treats as the fast-moving index, which the existing template's split_d_inner/d_conv structure can accommodate via a compile-time or runtime layout switch without restructuring the outer block/thread grid (block.x = sequence, block.y = channel-chunk, thread = channel-within-chunk stays the same either way; only the inner x[j]/w[j] gather addressing changes). The CPU reference must be added first as the correctness oracle before the CUDA path, per this project's standard op-implementation order.

## Code Samples & Guidance (historical sketch)

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

2026-09-24 GPT review req_e17e0bf5a68c48d5 applied: replaced the implicit-default layout-parameter approach (which violated this project's no-legacy-shim doctrine) with a separate ggml_ssm_conv_ext() function; defined exact channels-major shape/index algebra (d_inner=sx->ne[0], n_t=sx->ne[1]-d_conv+1) for CPU and CUDA/HIP; required either full backend coverage or capability-aware graph-selection before switching callers.

2026-09-25 (4d7117f6): implemented as patch 1263_prbe41_ssm_conv_channels_major, a port of nasone/AMD 33611a98 generated with bigcherry.patch.port_diff (17 files: ggml.h/ggml.c op mode, CPU + CUDA/HIP ssm-conv kernels, other backends' supports_op rejection, qwen35/qwen35moe/qwen3next graph, delta-net-base build_conv_state hand-merged onto b11126, fork test-backend-ops cases). Metal rejection not ported; conv-state layout changes (saved states not interchangeable). Activation marker patch=1263_prbe41 path=ssm_conv_channels_major. Contract PRBE41-SSM-CONV-CHANNELS-MAJOR (owner policy, 4 sessions/arch): full-vocab backend reference on qwen4b, pp512 positive, gpt-oss tg128 control. Rebase check CLEAN on bigcherry and bigcherry-tuning. Hardware sessions queued after the current batch.

## 2026-10-08 b11474 experiment scope

**Rank 7, qualify existing 1263 rather than re-port.** `1263_prbe41_ssm_conv_channels_major` state=untested (patch.toml); b11474 composed targets `ggml/include/ggml.h::ggml_ssm_conv_ext`, `ggml/src/ggml-cuda/ssm-conv.cu`, `src/models/delta-net-base.cpp` and Meta axis0 split/reshape propagation. Input channels-major shape `[d_inner,n_t+d_conv-1,n_seq]`, output `[d_inner,n_t,n_seq]`. Current Meta/tensor-split test failed `GGML_ASSERT(src_ss[0].nr[0] == 1)` before activation (PRBE114 evidence), so qualify single-GPU GDN first and treat multi-GPU as blocked until split-state instrumentation fixes actual axis ratios. Env `BIGCHERRY_SSM_CHANNELS_MAJOR=0|1` proposed opt-in for qualification if existing 1263 has no reliable selector; no implicit legacy default args. Expected gain uncertain (memory coalescing), only on GDN/27B SSM_CONV, not Flash-Next attention path. `queue-env-ab.sh`: 27B GDN 8K/24K/98K and single-gpu/tensor-split arms, output+recurrent-state parity, PPL, capture/replay, marker, VRAM, separate source-split traces. Do not claim multi-GPU validated.

Recheck every historical b11126 anchor on the composed b11474 source before coding; no GPU run or patch implementation is claimed by this plans-only triage.

## Change Log

- 2026-10-08 (triage): Kept experiment pending (P3); source/shape, coverage against #29901/1202/1253, env gate, expected effect and separated hardware test defined above.

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
- 2026-09-24T05:07:59.505020+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-09-25T10:49:09.867182+00:00 (updated-by): Updated: section:notes
- 2026-09-25T10:49:12.783952+00:00 (state-transition): State: pending → in_progress

## 2026-10-14 evidence/route reconciliation (authoritative over earlier historical wording)

**1263 remains `untested` in `patch.toml` for the current production pin; historical validated receipts do not automatically promote it.** The current `patch.py` explicitly gates all three Qwen hybrid model call sites on `model.split_mode() != LLAMA_SPLIT_MODE_TENSOR`. Tensor split builds the original time-major graph, emits `ssm_conv_channels_major_declined_split` when tracing, and must **not** be required to emit the positive channels-major marker. The b11233 pre-guard tensor-split crash is historical, not the expected behavior of the guarded patch.

The existing `evidence/validation.json` includes contract-passing, single-device `tierA-qwen4b-q6k` pp512 sessions: gfx1100 b11233 approximately +1.13% to +1.43%; gfx1201 b11126 +2.37% and +2.58%. Correctness receipts report zero full-vocabulary logprob difference; `tierM-gptoss20b-q6k` tg128 is a non-SSM control. Do not conflate these with dual-XTX Qwen3.8-27B `-sm tensor`: `docs/evidence/lab-run-summaries/ab-27b-1263.json` reports only +0.14% pp4096, and the current guarded route does not remove its transpose. All are historical measurements on older pins, not new b11552 GPU qualification.

**Ownership:** PRBE41/1263 retains single-device channels-major kernel/layout and contract ownership. PRBE114/BCOP195 closes the unqualified generic Meta RESHAPE/VIEW tensor-split rewrite; no second scheduler or split-state cache. Preserve the mode-specific conv-state lifetime, explicit fallback, full-vocab/MTP correctness, and current-pin composition/activation checks before any future promotion.


## 2026-10-15 BCOP234: route-bound activation and prefill qualification (authoritative)

**Disposition:** Patch 1263 remains `untested` on b11552. This section supersedes the historical `ggml_ssm_conv_ext` and default-argument sketches below: the actual implementation already uses `ggml_ssm_conv_channels_major`. Do not add another layout selector, scheduler, cache, allocator, or telemetry system. No new GPU campaign is authorized until activation evidence is repaired.

### Code-level finding: false-positive activation contract

`engines/llamacpp/patches/1263_prbe41_ssm_conv_channels_major/patch.py::_CALL_NEW` sets `channels_major = model.split_mode() != LLAMA_SPLIT_MODE_TENSOR` in `qwen35.cpp`, `qwen35moe.cpp`, and `qwen3next.cpp`. `_TIME_MAJOR_FALLBACK` preserves original RESHAPE/TRANSPOSE/CONCAT(axis 0), state views and copy, and emits `path=ssm_conv_channels_major_declined_split`. However `validation/producer.py::_MARKER_REGEX` and `validation.toml::activation.marker-regex` both match the **unbounded prefix** `path=ssm_conv_channels_major`. The producer's `re.search` accepts the decline marker; with no control hit, `ActivationEvidence(status="executed")` and the promotion trigger can pass on a time-major fallback. This is a demonstrated source/host **prospective evidence defect**, not a reproduced GPU failure or retroactive invalidation of older correctly routed single-device receipts.

The positive marker is emitted in `ggml/src/ggml-cuda/ssm-conv.cu::ggml_cuda_op_ssm_conv` before successful kernel completion, guarded by `std::once_flag` (process-once). The correctness server and separately measured `llama-bench` processes do not share an execution witness. The graph's channels-major `sx=[d_inner,n_t+d_conv-1,n_seq]`, recurrent state `[d_inner,d_conv-1,n_seq]`, output `[d_inner,n_t,n_seq]` and CONCAT(axis 1) use CPU `chan_nb/time_nb` and HIP short/long `stride_xc/stride_xt` addressing. Tensor-split remains time-major with CONCAT(axis 0). Layout must be fixed for the model/context/graph/state lifetime, including MTP rollback and saved-state restore.

### Minimal implementation-ready corrective gate (PRBE41/1263 owner)

1. Change **both** existing regexes to `BIGCHERRY_PATCH_HIT patch=1263_prbe41 path=ssm_conv_channels_major(?![A-Za-z0-9_])` (preserve TOML escaping). Parse `ssm_conv_channels_major_declined_split` separately. Positive admission: `subject_positive && !control_positive && !subject_declined`. Tensor-split negative control: `subject_declined && !subject_positive`. Mixed routes fail closed as `AMBIGUOUS_ROUTE`; never substring-match. No new telemetry.
2. Extend `tools/tests/patch/test_1263_prbe41_ssm_conv_channels_major.py` or patch-local producer tests to exercise the actual TOML and Python patterns on positive, declined, absent, wrong-patch, suffix, both-markers, control-only and repeated same-process logs. The process-once marker cannot prove every request; correlate subsequent requests with existing graph/dispatch traces.
3. Before performance attribution, establish that the **benchmarked** HIP subject launches `ssm_conv_f32<...,channels_major=true,...>` on the declared GPU UUID and has no CPU SSM_CONV fallback. Use existing rocprofv3/graph evidence for per-ubatch CONCAT/CONT nodes, launches, bytes, duration, stream synchronization, capture/replay and removed transpose. A server marker without a benchmark-side witness is `UNPROVEN_EXECUTION`.
4. Direct oracle: CPU time-major vs channels-major vs HIP, `d_conv=3/4/9`, `d_inner=128/256` and actual model width, `n_t=1/2/128/512/2048`, `n_seq=1/>1`, short/long kernels, contiguous strides, same-process multi-request/multi-ubatch, 8K/24K/98K, full-vocab logits/KLD, greedy/MTP acceptance, state rollback, memory safety and graph replay. Tensor-split must decline and preserve the original graph; non-SSM gpt-oss tg128 and gfx1030 are negative controls.
5. Only if source/route/correctness gates pass, compare composed b11552 baseline vs +1263 on single gfx1100/gfx1201, identical quant, batch/ubatch, model, context, clocks and device. Start with pp512; pp2048/4096 only when traces show exposed CONCAT/CONT headroom. Four independent order-balanced sessions/architecture, >=10 paired ABBA rounds/session; CI95-low >=3% E2E prefill, <=1% decode/control regression, exact work/bytes and numerical/state parity. Reject/default-off if removable E2E ceiling <3%, execution unproven or any state failure.

### Existing evidence, upstream and ownership

Historical `tierA-qwen4b-q6k` pp512 receipts: +1.13%..+1.43% gfx1100 (b11233), +2.37%/+2.58% gfx1201 (b11126), zero reported full-vocab logprob difference; not b11552 results. Dual-XTX Qwen3.8-27B tensor-split pp4096 +0.14% is fallback, not channels-major acceleration. [AMD-Ecosystem #52](https://github.com/AMD-Ecosystem/llama.cpp/pull/52) merged 2026-07-16 with external gfx1151 ubatch-dependent gains. New upstream [llama.cpp #30591](https://github.com/ggml-org/llama.cpp/pull/30591), open 2026-10-15, instead fuses a **Vulkan** transposed-input CONCAT via a padded 32x33 shared-memory tile; external R9700 Vulkan results do not justify a HIP port. [vLLM ROCm GDN](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py) distinguishes dimension-first/sequence-first conv-state and uses AITER fused decode, a state-lifetime reference rather than GGML replacement.

PRBE41/1263 owns single-device layout and validation. PRBE114's generic Meta tensor-split rewrite remains closed; PRBE18/PRBE21 own distinct qkv/CONCAT fusion. Do not touch active MEN10/RHP/Radiance.
