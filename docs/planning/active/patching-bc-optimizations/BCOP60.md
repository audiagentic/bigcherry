---
id: BCOP60
order: 60
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-08T16:09:26+11:00'
created-by: agent
priority: P2
---

# RNX10: qualify scalar shared-expert gate tail before fusion

## Discovery and change

Pinned llama.cpp b11474 already fuses shared up/gate/GLU and same-shape SIGMOID+MUL. Qwen4Exp's shared gate [1,M] versus shared output [H,M] prevents the latter matcher from applying; matmul-scale fusion is NVFP4-specific. RNX10's historical proposed patch 1310 collides with production `1310_act_q81`. No RNX10-specific kernel or E2E gain has been measured.

## Ownership and protected work

RNX10 owns only the scalar tail; RNX09 owns promotion. RNX04/PRBE14/QFP35/1215+1216/QFP18 own HC/router, routed weights, GLU, stream overlap and activation cache. RNX10 last changed independently on 2026-10-03 15:24 UTC; recent QFP43/MTP, PRBE triage, patch/CI and expert-backend work is protected and untouched. No RNX10 implementation, PR or queued lane was found.

## Bounded disposition

Gate 0 source/host shape checks passed. Gate 1 uses existing rocprof/graph diagnostics; if removable tail <5% E2E wall or graph/rank ownership unproven, close without patch. Otherwise one guarded existing-CUDA-fusion extension, exact native fallback and correctness/graph/rank tests. Promotion requires ≥10 paired rounds, CI95-low ≥3% E2E and ≤1% control regression. Do not allocate 1310 or duplicate native GLU/GEMM fusion.

## Sources and validation boundary

https://github.com/ggml-org/llama.cpp/blob/b9acf138/ggml/src/ggml-cuda/ggml-cuda.cu
https://github.com/vllm-project/vllm/issues/43187
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen2_moe.py

External MI355x/AITER evidence is not BigCherry RDNA3/4 performance. Existing BigCherry 1215+1216 and 1207 results are different mechanisms. Eight source assertions and four host shape fixtures passed; no build or hardware lane ran.
