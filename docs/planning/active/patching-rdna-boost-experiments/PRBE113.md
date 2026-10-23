---
id: PRBE113
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-29T23:14:00.745535+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Dual-XTX Q8_0 decode: retire repeated width-6 profile; qualify residual width census

## Authoritative disposition (2026-10-13; BCOP180)

**No new patch or broad GPU campaign.** The original three-measurement P0 plan is superseded. PRBE113 owns only receipt-first, per-width production decode profiling. Patch 1245's width-6 fused-vs-unfused profiling **already ran** and was rejected. Its n_max=5/verify-width-6 trace does not represent n_max=4/widths 2-5. RD13/1206 activation work is blocked by PRBE12/BCOP147's current b11552 RESHAPE/VIEW geometry safety issue; do not duplicate its counter or producer.

## Source and measured evidence

- `1245_gp11_mmvq_fusion_ncols_gate/patch.py` widens `mul_mat_vec_q_switch_fusion` fused instantiation only for Q8_0 ncols=6 and `ggml_cuda_should_fuse_mul_mat_vec_q` for dense gfx1100. MoE MUL_MAT_ID and MMVF remain excluded. Pinned llama.cpp b11552 and current upstream `ggml/src/ggml-cuda/mmvq.cu` still instantiate fusion only when `c_ncols_dst == 1` (same inspected blob SHA).
- **Historical dev measurement** (`1245/SUMMARY.md`): n_max=5, dual gfx1100, Qwen3.8-27B Q8_0: 112,110/123,090 MMVQ dispatches were unfused width 6; 1,120 were fused width 1. Actual 1245 A/B removed 8,450 quantize and 8,450 GLU calls, but MMVQ summed kernel time rose 3470.4 -> 4090.8 ms (+620.4 ms); quantize+GLU saved only 22.4 ms; net summed busy penalty ~598.0 ms. Single-run throughput fell 51.54 -> 46.70 t/s (-9.4%). Fused width 6 used 60 VGPR/0 scratch versus 44 VGPR unfused. The cause (occupancy, locality, live accumulators) is **not proven**. These are real historical, non-contract-qualified measurements, not new results or E2E kernel attribution. Do not rerun unchanged 1245.
- `tools/bigcherry/profiling/rocprof.py::parse_kernel_trace` already aggregates kernel-name calls, busy us, VGPR, scratch and GPU-agent sets. It **loses per-dispatch ordering, phase and physical rank**. `rocprofv3_command_prefix` provides sys/RCCL traces; `build_gpu_profile_pass` checks multi-GPU completeness. Reuse existing `tools/lab/flash-next/kernel-step-census.py`, profile scripts and receipts; no new profiler service or queue framework.
- PRBE115's `1301_prbe115_q8_f32_mtp_widths` is rejected for MTP acceptance/numerical drift; do not widen F32 activation again. PRBE111 owns IQ4_XS/IQ3_XXS and needs independent IQ evidence. PRBE12 owns 1206 and its shape-safe fusion producer.

## Gate 0: cheapest completed-receipt discriminator

Search existing **terminal** rocprofv3 receipts before hardware. Require exact Qwen3.8-27B Q8_0 model digest, llama.cpp pin/build/patch manifest, two physical gfx1100 UUIDs and rank order, tensor split, ROCm/driver, context, batch/ubatch, KV/FA, graph state, fixed draft source, n_max=4, steady decode window, token/acceptance accounting, trace timestamps and profiler completion. A filename/flag or queued job does not prove executed width.

Parse per-dispatch `(Agent_Id,Stream_Id,Kernel_Name,Start_Timestamp,End_Timestamp)`; map agent to physical UUID, extract MMVQ `(type,ncols_dst,has_fusion,small_k,halve_iters)`, and report **per rank and width 1..5** calls, busy us, median/p95, VGPR/scratch, Q8_1 quantize, GLU, GDN, attention, AllReduce, copy/CONT, CPU scheduler and launch gaps. Exclude warmup; identify unparsed kernels. Reject missing physical identity, phase, fixed-work, pin, depth or completeness as INELIGIBLE. Summed concurrent GPU busy time is not E2E wall time or a removable critical-path share.

If matched evidence exists, publish the per-width table here and stop. Otherwise allow **one stock host-exclusive** n_max=4 capture only after active Radiance/helper and other queued hardware work is terminal and devices are idle. No 1245 variant, extra profiler or competing GPU queue.

## Gate 1: bounded terminal decision

Rank each width's **exposed E2E** Q8_1 quantize+MMVQ+GLU cost against communication, host gaps and GDN. If optimistic E2E removable ceiling <3%, close candidate without kernel work. If >=3% and genuinely measured, hand only the exact dtype/width/shape/architecture/model/receipt to PRBE115 for a separate numerics-preserving design. Never equate eliminated launches with speedup (1245 removed 25,350 but lost). PRBE12 independently clears 1206 geometry safety first.

Any subsequent promotion requires same physical devices/model/pin/work, full-vocab logits/KLD, deterministic greedy, MTP draft/acceptance parity, multi-request, multi-ubatch, long context, graph replay, four independent sessions and >=10 order-balanced paired rounds each; CI95-low >=3% E2E and <=1% unaffected-control regression. Missing work or transfer is a failure.

## Upstream and external comparison

Inspected llama.cpp [PR #29948](https://github.com/ggml-org/llama.cpp/pull/29948): new gate/up/GLU **MMQ prefill** fusion explicitly rejects `ggml_cuda_should_use_mmvq` and gates on `turing_mma_available` (NVIDIA-only). Related [#28702](https://github.com/ggml-org/llama.cpp/pull/28702) is also MMQ, not a proven AMD MMVQ substitute. Reuse only phase/shape fail-closed admission. [vLLM gfx1100 W4A16 GPTQ fused MoE](https://github.com/vllm-project/vllm/blob/main/csrc/rocm/moe_q_gemm_rdna3.cu) is a different weight/routing path; reject as dense Q8_0 MMVQ replacement.

## Validation and history

Standalone synthetic CSV/historical arithmetic fixture: **10/10 passed** (rank/width grouping, malformed cohort rejection, historical +598.0 ms net summed-kernel penalty and -9.4% throughput). Not repository pytest, real rocprof, HIP build or GPU run. No new measured gain. Last independent PRBE113 plan change 2026-10-08; no independent PRBE113/1245 work within preceding 12 hours. Recent Radiance expert-helper/RAD01 work is excluded and untouched.

2026-09-29 original P0 three-job plan; 2026-09-30 1245 rejected; 2026-10-04 1301 rejected; 2026-10-12 PRBE12/BCOP147 safety gate; 2026-10-13 BCOP180 replaces duplicate capture with receipt-first bounded decision.
