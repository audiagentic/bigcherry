---
id: BCOP191
order: 191
plan: patching-bc-optimizations
state: done
created-at: '2026-10-13T18:12:28Z'
created-by: agent
priority: P3
work: S
---

# QFP48: gate Q8_1 MMQ producer fusion on typed route and exposed cost

**Discovery/change:** Pinned b11552 `ggml_cuda_mul_mat_q` quantizes F32 inputs into weight-type-dependent MMQ Q8_1 layouts (D4/DS4/D2S6), with separate routed `ids`/scatter and FP4 paths. Validated 1307's producer/cache is MMVQ-specific, not an MMQ prefill buffer. The old QFP48 `quantize_mmq_q8_1 -> mul_mat_q` adjacency is not sufficient to justify a generic fusion. Historic ub256 figures give only a **2.86% illustrative per-step perfect-removal bound**; not an E2E measurement. Upstream #29948 is NVIDIA-only dense gate/up/GLU, not AMD routed-MoE evidence; vLLM's ROCm FP8 producer fusion supplies only typed matching as a reference.

**Owner and exclusions:** QFP48 retains the technical contract and remains pending for other trace-led fusions. 1307/QFP13 own existing Q8_1 MMVQ producers and cache; 1313/1355 own their current gate fusions. QFP16/QFP50 own existing input/pipeline evidence; no edits or new queue under them. No independent QFP48/MMQ-producer activity in the preceding 12 hours; current Radiance mailbox/helper, patch-prefix/release, and associated hardware work excluded. Recent BCOP181-190 do not cover this MMQ-specific candidate.

**Unresolved action:** None to implement now. Reopen **only** after an exact-revision, per-device matched trace proves a single-consumer F32 dense MMQ producer, exact D4/DS4/D2S6 output contract, no routed/scatter/FP4 ambiguity, and >=3% perfect-removal E2E ceiling. Then reuse existing MMQ/1307 ownership for one default-off bounded experiment; require numerical, graph, request-lifetime and AMD ABBA gates in QFP48. Otherwise terminal **reject** this fusion candidate without another cache/dispatcher/allocator.

**Validation/evidence:** llama.cpp b11552 `ggml-cuda.cu`, `mmq.cu`, `quantize.cu`; BigCherry 1307 SUMMARY; QFP48 trace notes; [upstream #29948](https://github.com/ggml-org/llama.cpp/pull/29948), [vLLM ROCm fusion](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rms_quant_fusion.py). Source assertions 14/14; disposable host selector tests 10/10; Python compile passed. No HIP compile, GPU run or new performance result.
