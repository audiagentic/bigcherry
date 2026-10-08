---
id: PRBE59
order: 0
plan: patching-rdna-boost-experiments
state: completed
created-at: '2026-09-09T10:57:36.613381+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# RD76 / FORK-HIP-001: retire obsolete rocWMMA FlashAttention restoration

## Decision (2026-10-08)

**Close without patch or hardware queue.** llama.cpp PR #26046 (merged 2026-07-24) removed the legacy rocWMMA FlashAttention kernel (`fattn-wmma-f16.cu`, 705 lines), header, HIP build flag and dispatch. Pinned b11474 already uses the maintained native AMD MMA kernel; upstream PR #28102 (merged 2026-09-11) further tuned that same path for gfx1201. The proposed new rocWMMA build option, translation unit and runtime selector would revive intentionally deleted code. The fork's 5aa2f049 was build plumbing only, not a kernel.

## Verified code and ownership

- `ggml/src/ggml-cuda/fattn.cu::ggml_cuda_get_best_fattn_kernel` (~541, ~700) selects `BEST_FATTN_KERNEL_MMA_F16` on eligible AMD WMMA GQA shapes with DKQ <=256 (special heads 40/72 excluded). `ggml_cuda_flash_attn_ext` (~732-741) dispatches it; no legacy WMMA enum/include exists.
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh::ggml_cuda_fattn_mma_get_config` owns head/row/tile/SMEM tuning, including DKQ=256; `ggml/src/ggml-cuda/fattn-common.cuh::launch_fattn` owns Stream-K. Do not duplicate the selector, kernel registry, allocator, graph or launch policy.
- `ggml/CMakeLists.txt` and `ggml/src/ggml-hip/CMakeLists.txt` contain no `GGML_HIP_ROCWMMA_FATTN`. `tests/test-backend-ops.cpp` has head-256 F16 and quantized-KV FA cases. Existing per-device ggml FA graph/lifetime and Q/K/V storage remain authoritative; non-P2P gfx1100/gfx1201/gfx1030 PCIe topology is not a reason to restore this kernel.
- Separate rejected `patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq` owns RD06 native WMMA config history. Its **BigCherry-measured** kernel delta was -0.0154% (CI95-low -0.0745%; gate >=0.5%); that is **not** rocWMMA performance evidence. RD07 Q6_K belongs to PRBE110. Vulkan scalar FA belongs to PRBE65; active MTP/patch-system paths are untouched.

## External evidence / mechanism

Upstream PR #28102 reports **external** gfx1201 Qwen 27B IQ4_XS 150K-context pp512 164.42±5.79 -> 399.01±30.87 t/s, tg128 19.70±0.20 -> 19.67±0.39. These are not BigCherry results. Issue #24961 reports a legacy rocWMMA gfx1201 long-prefill HIP-graph UpdateStreams crash near 106K, avoided with graphs off; it does not prove the precise root cause. vLLM uses Triton attention for gfx11/gfx12, and AITER labels gfx1100/gfx1201 experimental; neither supplies a drop-in GGUF rocWMMA replacement or measured local improvement.

Sources: https://github.com/ggml-org/llama.cpp/pull/26046 ; https://github.com/ggml-org/llama.cpp/pull/28102 ; https://github.com/ggml-org/llama.cpp/issues/24961 ; https://github.com/ROCm/aiter ; https://docs.vllm.ai/en/v0.16.0/api/vllm/platforms/rocm/ .

## Terminal / conditional reopening

No next RD76 implementation gate. A *new owner* may reopen only for a maintained genuinely different kernel after first-party profiling proves >=5% E2E wall time in eligible native FA shapes. First perform static architecture/shape/resource/graph-lifetime checks; only then run >=4 paired sessions per gfx11/gfx12 lane with pp512/2048, tg128, 32K/64K/160K, multi-request/multi-ubatch, F16/quant KV, greedy/logits/KLD, MTP acceptance, memory safety, graph on/off and actual-vs-expected FA work/transfer counts. Require CI95-low >=3% E2E gain and <=1% control regression. Keep native fallback; no new config/selector unless that gate passes. Otherwise reject and remove disposable variants.

## Audit validation and provenance

11 static assertions against upstream b11474 passed (old flag/header/enum absent; native selector/config/dispatch and head-256/quant-KV tests present). No build, backend-op execution, prototype or hardware benchmark. Original plan created 2026-09-09, last independently changed 2026-09-24 (39b146c7). Historical proposals for new rocWMMA FA are superseded, not implemented. BCOP59 is the thin disposition ledger.

## Change Log

- 2026-09-09T10:57:36.613381+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:47.897740+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.393793+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.204866+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:13:33.193064+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031346_repaired-four-more-migrated-pa_4345
- 2026-09-10T03:13:46.522718+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:52:43.672823+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-24T04:53:46.389230+00:00 (updated-by): Updated: section:notes
- 2026-09-24T05:09:39.917881+00:00 (updated-by): Updated: section:description, section:steps, section:notes
