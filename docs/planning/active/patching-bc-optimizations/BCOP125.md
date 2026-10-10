---
id: BCOP125
order: 125
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-10T22:11:49+00:00'
created-by: agent
priority: P2
work: S
---

# PRBE111: reject unproven IQ MMVQ hoist; preserve 1273 ownership

## Discovery / disposition

Pinned b11474 IQ4_XS computes scale after its four-iteration dot loop; IQ3_XXS loads aux32 before its loop and uses iteration-specific sign/grid inputs. The proposed block-invariant hoist has no demonstrated repeated work. Two vec-dot/VDR switches are constexpr (23 explicit vec-dot types, 21 explicit VDR types; IQ1_S/M use default VDR1), not proven runtime dispatch overhead. Keep PRBE111 pending **only for an IQ-specific trace/codegen discriminator**; no speculative kernel or descriptor refactor. Merged upstream #27962 SWAR is already in the pin. Open upstream #29025 tail-row clamp is a conditional correctness control for rows-per-block >1, not a proven IQ single-token defect.

## Ownership / 12-hour exclusion

PRBE111 owns the IQ instruction-level decision; 1273 owns IQ4_XS/IQ3_XXS VDR/nwarps tuning, PRBE47 Q6_K reductions, PRBE113 Q8_0 profiling methodology (not IQ attribution). Last independent PRBE111 change: 2026-10-09 05:35 UTC; 1273 last path change: mechanical relocation 2026-10-09 04:20 UTC. No IQ-specific implementation, submission, queue or PR found within the preceding 12 hours. Active QFP49/1360 transfer-stream, QFP50/1359 prefill, QFP17/1330, QFP41/1356, QFP36/1357 and Radiance development remain excluded and untouched.

## Bounded next action / terminal outcome

Consume existing traces/queues first. If a real IQ4_XS gfx1100/gfx1201 lane is available, capture stock `mul_mat_vec_q` kernel/E2E, ISA and resources before editing. Reject specialization when removable exposed latency <2.9126% E2E or no repeated instruction is proven; otherwise qualify a single default-off, composed 1273-safe helper with exact numerics and >=3% CI95-low E2E, <=1% control regression. If no IQ trace exists, record BLOCKED_ON_IQ_TRACE without inventing a speedup or new queue. Detailed code/test plan lives only in PRBE111.

## Evidence

Pinned `mmvq.cu`, `vecdotq.cuh`, `vendors/hip.h`; 1273 patch.py/SUMMARY; [llama.cpp #27962](https://github.com/ggml-org/llama.cpp/pull/27962), [#29025](https://github.com/ggml-org/llama.cpp/pull/29025), [SGLang GGUF dispatch](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/gguf.py), [vLLM GGUF plugin MMVQ](https://github.com/vllm-project/vllm-gguf-plugin/blob/main/vllm_gguf_plugin/csrc/gguf/mmvq.cuh). 9/9 source checks, 20,000 SWAR host pairs, 51 row-tail host fixtures. No HIP build, pytest, GPU benchmark or new measured gain.
