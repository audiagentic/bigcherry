---
id: BCOP133
order: 133
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T07:10:00+00:00'
created-by: agent
priority: P3
work: S
---

# RNX05: retire duplicate MXFP4 type and occupied patch IDs

## Discovery / change

Pinned llama.cpp b11474 already defines `block_mxfp4` (32 weights/17 bytes), `LLAMA_FTYPE_MOSTLY_MXFP4_MOE`, CUDA/HIP MXFP4 MMVQ/MMQ and RDNA2/3/4 configs. The old RNX05 proposed new type/1305+1306 ports; those IDs now belong to production FFN split and shared-expert split. The r9700-stack FP8 WMMA kernel uses a load-time fragment layout and sorted-token contract that cannot alias GGUF's interleaved blocks. Native fallback is the control; no incremental BigCherry speedup is established.

## Owners / novelty / protected work

RNX05 owns the narrow FP8-activation candidate disposition; RNX01 source intake; RNX09 integration. Existing 1237/1265 compact grid, 1273 IQ tuning and 1274 Q6_K decode remain authoritative. Merged upstream #29809 provides SYCL arithmetic decode and bounded lazy expert reorder; its Intel gains do not qualify AMD. No independent RNX05/RNX01/1237 work was found in the preceding 12 hours on inspected active branches; recent BCOP124–132 cover other capabilities. Active QFP41/1356, QFP49/1360/1361 and QFP48 remain protected; no code/patch/hardware queue changed.

## Bounded action / terminal disposition

Use an existing MXFP4 GGUF to prove actual native dispatch, expert shape and exposed kernel cost; run format/inverse-map host fixture first. Reject if no deployable MXFP4 lane, non-equivalent layout, or <2.9126% removable E2E time. Only then qualify one gfx1201-only default-off FP8-activation prototype against the *same model*, preserving native MMVQ/MMQ. Require logits/KLD/MTP/replay correctness and CI95-low >=3% E2E gain with <=1% control regression. Otherwise close `ALREADY_SUPPORTED_NATIVE`. Technical detail and test matrix belong exclusively to RNX05.

## Evidence / validation

Pinned b11474 `ggml-common.h`, `llama-quant.cpp`, `ggml-cuda.cu`, `mmq.cu`, `mmvq.cu`, `mmq-config-rdna{2,3,4}.cuh`; r9700-stack `r9k_moe_mxfp4a8.hip`, `r9k_gemm_fp8.hip`, `r9700_vllm/kernels/moe.py`; merged upstream #29809 diff and open #28709. Source assertions **11/11** and illustrative two-block AoS/SoA host fixture passed. No HIP build, GPU execution or new hardware benchmark. External SYCL 1.91× decode is not AMD evidence.
