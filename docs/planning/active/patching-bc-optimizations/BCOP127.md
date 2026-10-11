---
id: BCOP127
order: 127
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T00:08:10+00:00'
created-by: agent
priority: P3
work: S
---

# PRBE18: retire 16-node SSM pre-scan premise; use native fusions

## Discovery / disposition

Pinned llama.cpp b11474 Qwen3.5 GDN pre-scan is a Q/K/V fan-out from SSM_CONV+SILU, with separate alpha/gate and beta producers. `build_gdn_l2_norm` emits RMS_NORM+SCALE; native HIP already fuses both SSM_CONV+SILU and RMS_NORM+SCALE. The historical 16-node linear matcher, shared alpha/beta norm and single-output skip plan cannot be admitted without an actual graph/output proof. Merged upstream #28068/#29393 supersede the old norm assumption. Retire the proposed 16-node port; retain PRBE18 only for a bounded residual-cost discriminator. This is source evidence, not a GPU regression or speedup.

## Ownership / exclusion

PRBE18 owns only a residual Qwen3.5 GDN pre-scan fusion decision. PRBE05/QFP13 and validated 1307 own Q8_1 MMVQ activation cache (1235 retired); PNRO04/05 own GDN recurrence; upstream #29187 separately owns alpha/beta producer fusion. Do not create another cache, scheduler or kernel family. PRBE18 last independent file update was 2026-10-08 06:47 UTC; 1307 last path update was mechanical relocation 2026-10-09 04:20 UTC. No PRBE18/SSM pre-scan implementation, PR, benchmark or queued hardware experiment was found in the inspected last-12-hour repository/PR evidence. Protected QFP49/1360/1361, QFP50/1359, QFP41/1356, QFP17/1330 and Radiance work was not changed.

## Bounded action / terminal gate

Consume existing Qwen3.5-27B GDN graph/rocprof receipts. Verify actual post-scheduler Q/K/V alias edges and native fusion hit counts; reject if no residual or <2.9126% exposed E2E wall, otherwise qualify exactly one narrower default-off candidate with full recurrent/MTP/graph/no-P2P correctness and >=3% CI95-low E2E, <=1% controls. Missing trace = BLOCKED_ON_GRAPH_TRACE; no new hardware queue while active lanes run. Technical design and tests belong in PRBE18.

## Evidence and validation

Pinned `src/models/qwen35.cpp::build_layer_attn_linear`, `src/models/models.h::build_gdn_l2_norm`, `ggml/src/ggml-cuda/ggml-cuda.cu::{ggml_cuda_can_fuse,ggml_cuda_try_fuse}`, `ggml/src/ggml-cuda/ssm-conv.cu::ggml_cuda_op_ssm_conv`, `1307_q81_activation_cache_mmvq/SUMMARY.md`; [llama.cpp #28068](https://github.com/ggml-org/llama.cpp/pull/28068), [#29393](https://github.com/ggml-org/llama.cpp/pull/29393), [#29187](https://github.com/ggml-org/llama.cpp/pull/29187), [vLLM #51779](https://github.com/vllm-project/vllm/issues/51779). 256 host float64 formula cases and 3 synthetic graph invariants passed. No repository pytest, C++/HIP build, real graph trace, hardware run or new performance measurement.
