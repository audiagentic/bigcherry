---
id: PRBE18
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:45.862958+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P3
---

# PRBE18 — Qwen3.5 GDN pre-scan fusion: reject stale 16-node port, qualify only residual work

## Authoritative disposition (2026-10-11, llama.cpp b11474)

**Reject the historical 16-node SSM_CONV→Q/K/V→gate/beta port as specified. No new patch or hardware job is authorized.** The actual Qwen3.5 graph is a *fan-out*, not the claimed single 16-op linear chain. Two of the most obvious opportunities are already native HIP fusions. Keep PRBE18 pending only for a bounded, read-only runtime graph/critical-path discriminator; close if it finds no independently removable cost. This supersedes the b11126 proposal, including its suggested `12xx_rd24` patch, 16-node matcher, raw historical warp geometry, and PRBE05/1235 dependency.

### Pinned source and exact data flow

1. `src/models/qwen35.cpp::graph::build_layer_attn_linear` (b11474, lines ~372–390): `beta` is `build_lora_mm(ssm_beta,cur) → RESHAPE → SIGMOID`; `gate` is `build_lora_mm(ssm_alpha,cur) → RESHAPE → ADD(ssm_dt) → SOFTPLUS → MUL(ssm_a) → RESHAPE`. **Both are independent inputs to GDN, not Q/K normalization outputs and not descendants of SSM_CONV.** No “shared alpha/beta between norm ops” exists.
2. Same function (~399–462): `build_conv_state` forms `conv_input`; `ggml_ssm_conv(conv_input,conv_kernel) → SILU`; the SiLU result has three distinct aliasing `VIEW_4D` consumers (`q_conv`, `k_conv`, `v_conv`) with different offsets. Q/K are separately normalized; V stays a view. `build_recurrent_attn` consumes normalized Q/K, V, gate, beta and recurrent state. The convolution-state update is separately published by `delta-net-base.cpp::build_conv_state`, including K rollback slots.
3. `src/models/models.h::build_gdn_l2_norm` (introduced by merged [#28068](https://github.com/ggml-org/llama.cpp/pull/28068), 2026-09-06) is `ggml_scale(ggml_rms_norm(x,eps/n),1/sqrt(n))`, **RMS_NORM → SCALE**, not an RMS_NORM-only node or a learned alpha/beta pair. In exact arithmetic this equals `x / sqrt(sum(x*x)+eps)`; floating-point evaluation/rounding still requires qualification.
4. `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_can_fuse/ggml_cuda_try_fuse` already recognizes and executes `SSM_CONV → SILU` (and `SSM_CONV → ADD → SILU`) via `ggml_cuda_op_ssm_conv`; it also recognizes `RMS_NORM → SCALE` via `ggml_cuda_op_rms_norm_scale_fused` (merged [#29393](https://github.com/ggml-org/llama.cpp/pull/29393), 2026-09-25). `ggml/src/ggml-cuda/ssm-conv.cu::ggml_cuda_op_ssm_conv` has **one output** (`silu_dst` or `dst`) and handles the conv/bias/SiLU only. It cannot silently replace Q/K/V branch outputs, their alias lifetimes or recurrent-state publication.
5. `ggml_can_fuse_subgraph` plus `ggml_cuda_check_fusion_memory_ranges` protect source edges, consumers and output aliases. A speculative matcher that skips 16 graph nodes while publishing only one destination would lose Q/K/V or unrelated gate/beta work. A graph-build order alone does not prove a contiguous runtime node window.

**Ownership correction:** former `1235_rd09_q81_activation_cache_foundation` was retired into validated `engines/llamacpp/patches/1307_q81_activation_cache_mmvq` (PRBE05/PRBE06/QFP13). That cache serves MMVQ's Q8_1 activation identity; it is **not a prerequisite for F32 SSM convolution or Q/K RMS_NORM**. Do not reintroduce 1235, import a second cache, or use MMVQ launch geometry to size an SSM_CONV kernel. PNRO04/PNRO05 own GDN recurrence/prefix-tail, not PRBE18. Active QFP49/1360/1361 prefill/Meta work remains excluded and untouched.

### External mechanisms and transfer decision

- Open [llama.cpp #29187](https://github.com/ggml-org/llama.cpp/pull/29187) separately fuses **alpha/beta projection producers**; it is not a conv→Q/K/V 16-node kernel. Inspect its final merged implementation and per-architecture evidence before considering an independent producer optimisation; do not absorb its ownership into PRBE18.
- [vLLM issue #51779](https://github.com/vllm-project/vllm/issues/51779) documents BF16 Q/K normalization/rounding sensitivity in fused GDN implementations. Its measurements are external and do not establish an AMD performance benefit; use it as a precision-parity warning.
- [SGLang KDA backend](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kda_backend.py) uses shape-gated fused decode with an explicit fallback; CUDA/Triton assumptions and state layout are not directly portable to gfx1100/gfx1201.
- Merged upstream [#30087](https://github.com/ggml-org/llama.cpp/pull/30087) changes GDN **recurrent** warp geometry, not the pre-scan. Do not count its reported NVIDIA benefit as PRBE18 headroom.

### Cheapest discriminator before any implementation

**Gate 0 — completed in this audit (host/source only):** b11474 source confirms the two native fusion paths, Q/K/V fan-out and independent gate/beta; 256 deterministic float64 equivalence cases for the normalized formula passed (max absolute difference 1.67e-16); three synthetic topology invariants passed. These are not compiled-graph, HIP or GPU observations.

**Gate 1 — read-only graph and timing evidence, not yet run:** consume existing Qwen3.5-27B Q8_0 receipts/rocprofv3/graph debug before scheduling work. For each GDN layer record the **actual post-scheduler** op list and `{node_id,op,src_ids,output_ids,shape,nb,type,device,stream,buffer,alias_root,consumer_count}`. Record which native `SSM_CONV+SILU` and `RMS_NORM+SCALE` fusions fired; count physical launches, Q/K/V materializations, bytes, graph capture/replay and per-rank kernel-time **union**, not sums of overlapping durations. Use single gfx1100 and gfx1201, dual gfx1100 tensor split/no-P2P only if an already scheduled qualified lane exists; decode n_tokens=1, MTP verify widths 2–5, pp512 negative/control, ctx 8K/98K. Keep active Flash-Next/QFP49 lanes untouched.

**Gate 1 terminal rule:** `REJECT_NO_RESIDUAL` if the native fusions activate and no separately removable materialization/launch is measured; `REJECT_NO_HEADROOM` if exposed removable critical-path time is <2.9126% of end-to-end wall (the theoretical latency reduction required for +3% throughput), or if branch alias/consumer requirements prevent elision. If the exact graph is not available, `BLOCKED_ON_GRAPH_TRACE`; do not fabricate a 16-node sequence.

**Gate 2 — conditional, only after a surviving trace:** identify **one** residual same-device op sequence and one output contract, e.g. a Q/K norm-specific layout conversion, with exact `ggml_cuda_try_fuse` matcher, edge/alias/stride/type guards and existing kernel/allocator ownership. No generic 16-node multi-output kernel, new GGML op, cache, scheduler or dispatch table. Keep native conv+SiLU, native RMS_NORM+SCALE and V aliasing unless the trace proves a narrower safe replacement. Default OFF and exact architecture/shape admission. Reject if additional persistent buffers, H2D/D2H or synchronization erase the exposed headroom.

**Gate 3 — correctness and performance:** host reference first (n=32/64/128/256, eps, zero/near-zero/subnormal, F32 and BF16-rounding controls), then backend-op and actual-model logits/KLD/greedy and MTP acceptance; state/rollback snapshots, all Q/K/V consumers, graph replay/reallocation, multi-ubatch and multiple requests in one process, negative wrong-edge/alias and mixed no-P2P fallback. Compare pristine b11474 against single default-off candidate, same model/quant/KV/context/VRAM, >=4 independent sessions/architecture and >=10 paired ABBA rounds; promote only if CI95-low >=3% E2E and no control regression >1%. Otherwise reject, retain negative evidence and close PRBE18.

### Provenance and historical correction

Historical RD24/PRBE18 proposed a 16-op fused SSM pre-scan, a `1235` Q8_1-cache dependency and a hard-coded matcher around b11126. The 2026-10-08 triage already made this trace-gated and separate from Flash-Next. This audit replaces its still-incorrect linear-chain, alpha/beta norm, stale cache and launch-count assumptions with b11474 source semantics. No patch, test suite, build or GPU benchmark was run as part of the audit.
