---
id: PRBE114
order: 0
plan: patching-rdna-boost-experiments
state: done
created-at: '2026-09-29T23:14:03.823307+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: S
---

# Tensor-split channels-major SSM_CONV: decline speculative Meta rewrite

## Authoritative disposition — 2026-10-14, pinned llama.cpp b11552

**DONE (current proposal retired), not a completed tensor-split implementation.** PRBE114 previously asked to change generic Meta RESHAPE/VIEW split propagation to enable 1263's channels-major GDN SSM_CONV in `-sm tensor`. The current 1263 package **already refuses that route**, and existing measurements do not establish an exposed tensor-split gain that justifies a cross-op split-state rewrite. Preserve the stock time-major path on tensor split; do not queue another GPU experiment or add a Meta scheduler/allocator/dispatch abstraction. Reopen only under the explicit evidence gate below. PRBE41 owns the existing single-device channels-major implementation and its separate qualification.

### Current implementation and correctness boundary

- `engines/llamacpp/patches/1263_prbe41_ssm_conv_channels_major/patch.py::_CALL_NEW` sets `channels_major = model.split_mode() != LLAMA_SPLIT_MODE_TENSOR` in **all three** `qwen35.cpp`, `qwen35moe.cpp`, `qwen3next.cpp` graph call sites. `_TIME_MAJOR_FALLBACK` restores `ggml_reshape_3d(..., d_conv-1, d_inner, n_seq)`, `ggml_transpose(qkv_mixed)`, time-axis-0 CONCAT, and original recurrent-state copy/view geometry. It emits `path=ssm_conv_channels_major_declined_split`; the channels-major activation marker is **not** expected in this route. This is a deliberate fallback, not proof that tensor split is optimized.
- In pinned `ggml/src/ggml-backend-meta.cpp`, `handle_reshape` computes a source split-axis product, then aborts on `GGML_ASSERT(src_ss[0].nr[0] == 1)` when output shape crosses the input split base. `handle_view` delegates contiguous views to that handler. `handle_ssm_conv` maps matching source axes and otherwise uses `handle_generic`. The 1263 patch also inserts a channels-major `handle_ssm_conv` arm (input axes 0/1 -> output axis 0), but **that does not repair an earlier RESHAPE/VIEW split-state mismatch**.
- Historical `docs/evidence/native-vs-patched-historical/ssm1263-activation/RESULT.md` records the **pre-fallback** b11233 dual-XTX startup assert, before any activation or benchmark. Do not misreport that old crash as the current guarded route.
- Conv-state layout changes between time-major and channels-major. Any future mode decision must remain fixed for the model/context/recurrent-state lifetime, including MTP rollback and graph replay. Reject cross-layout saved-state reuse and unsafe fallback after partial state writes.

### Existing measured evidence — not new measurements

- `docs/evidence/lab-run-summaries/ab-27b-1263.json`: Qwen3.8-27B Q8_0, dual gfx1100, `-sm tensor`, MTP4, four paired observations per arm. `pp4096` 1246.40 -> 1248.14 t/s, reported +0.14% [0.03%, 0.24%]; `pp1024` +0.05% [-0.11%, 0.27%]; acceptance 84.29% both. Because the current patch declines channels-major for tensor split, **these small differences cannot be attributed to removed transposes**. The receipt does not prove the active route without correlated marker/graph evidence.
- `engines/llamacpp/patches/1263_prbe41_ssm_conv_channels_major/evidence/validation.json` contains historical **validated** single-architecture contract records for `tierA-qwen4b-q6k` pp512: gfx1100 approximately +1.13% to +1.43% in b11233 four-session campaigns; gfx1201 +2.37% and +2.58% in separate b11126 campaigns. `tierM-gptoss20b-q6k` tg128 is a non-SSM control. Full-vocabulary correctness records report zero logprob difference. These are not dual-XTX tensor-split, not b11552 qualification, and not evidence of a >=3% production tensor-split opportunity. Some earlier records are incomplete or withdrawn; cite only contract-passing cohorts.
- AMD-Ecosystem llama.cpp PR #52 reports a much larger gfx1151 pp4096 gain for its different single-GPU fork/configuration. That is external evidence only; do not extrapolate to BigCherry gfx1100/gfx1201 tensor split.

### Upstream/fork reconciliation

- Upstream llama.cpp PR [#23843](https://github.com/ggml-org/llama.cpp/pull/23843), merged 2026-05-30, fixes Qwen3.5/3.6 mixed-quant **tensor granularity selection** for multi-GPU Meta. It is already part of b11552 and is **not** a channels-major RESHAPE/VIEW fix. Its associated [issue #22817](https://github.com/ggml-org/llama.cpp/issues/22817) is a different failure and must not be counted as 1263 validation.
- [AMD-Ecosystem PR #52](https://github.com/AMD-Ecosystem/llama.cpp/pull/52) implements the alternative layout in CPU/CUDA and a channels-major SSM Meta handler, but no demonstrated safe generic reshape propagation for this BigCherry tensor-split case. [vLLM ROCm GDN/AITER](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py) uses a different fused conv/state representation; do not port its allocator or scheduler.

### Bounded reopening/implementation gate (not currently authorized work)

1. **Receipt-first:** find an eligible current-pin dual-XTX Qwen3.8-27B tensor-split GDN trace with exact model/patch/driver/physical UUIDs, steady prefill ubatch, kernel timeline, `CONT` transpose launches/bytes, per-rank idle/critical path and explicit `declined_split` marker. If no eligible receipt, do not queue work while Radiance/helper or other experiments are active.
2. **Reject cheaply:** calculate a *measured exposed* upper bound for eliminating the specific transpose and necessary reshape/copies; if optimistic E2E improvement <3%, terminal reject. Summed kernel busy time or external gfx1151 results are not the bound.
3. **Only if >=3%:** build a disposable graph-only fixture (no GPU benchmark) for `qwen35`, `qwen35moe`, `qwen3next`. Capture per-node `(name, op, ne, nb, src axes/nr/segments/per-device split_ne)` at `build_rs -> RESHAPE -> CONCAT -> SSM_CONV -> VIEW/CPY`. Test two gfx1100 ranks with equal/unequal splits and three-rank mixed-quant controls. Fail closed at the *first* inconsistent split; do not suppress `nr[0]==1` or force `MIRRORED` without a byte/shape proof.
4. **Conditional implementation:** change the minimum verified Meta split-state propagation under existing `handle_reshape`/`handle_view` and `handle_ssm_conv` ownership; preserve original ratios/segments and per-rank tensor byte extents. Keep 1263's current tensor-split time-major path as default-off fallback. No new selector, cache, allocator or transport.
5. **Qualification:** graph-only invariant tests before HIP compilation; backend-op parity for both layouts and CPU/Meta, multi-request/same-process, multi-ubatch, long-context, recurrent-state/MTP rollback and deterministic greedy/logits/KLD, no missing/duplicate work, no graph allocation churn. Then >=4 independent order-balanced sessions per architecture, >=10 paired rounds each, CI95-low >=3% exposed E2E improvement and <=1% unaffected control regression. Rejection restores original fallback without mutating recurrent state.

**Terminal decision now:** no current tensor-split rewrite, no queued benchmark, no production promotion. Reopen only with genuinely new independent evidence satisfying gates 1-2. The single-device PRBE41 patch is separately owned and not retired here.

## History

2026-09-29 original plan requested Meta reshape propagation after b11233 startup assert. 2026-10-08 triage retained pending but did not reconcile 1263's already-present tensor-split guard or the validated single-device evidence. 2026-10-14 audit reconciles both and retires the speculative multi-device extension.
