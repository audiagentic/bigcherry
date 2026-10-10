---
id: PNRO02
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:07.096423+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P3
---

# Residual ADD in internal AllReduce finish — disposition and bounded re-entry

## 2026-10-10 source audit / decision

**Pending, blocked on an eligible production provider; no new patch or hardware queue.** The 2026-09-24 opening text claiming no matcher/API is obsolete: on 2026-09-25 those were implemented inside `1250_nro01_allreduce_q8_wire` and the 1251 scaffold was removed. The current 1250 package is **evaluated**, not validated. Its `patch.toml` requires `1252_nro03_allreduce_p2p_provider` (**rejected** after the 2026-10-07 no-P2P hardware fault) and `1272_ar_host_compressed_wire` (**untested**). Do not promote, compose, enable, or queue 1250 on the present topology merely to exercise PNRO02. BPB01's 2026-10-08 assessment already calls 1250 superseded/infeasible as a package; this audit resolves the remaining authoritative PNRO02 contradiction.

**No exact-F32 fused path exists in 1250.** `ggml_cuda_ar_allreduce_fused_add` in `engines/llamacpp/patches/1250_nro01_allreduce_q8_wire/patch.py` rejects any `p->wire_override != q8_0`, requires two devices and a positive copy threshold, quantizes each F32 partial to Q8_0, then calls 1272's Q8 finish with the mirrored residual. The earlier requirement “exact-F32 fusion independent of Q8” is an unimplemented *future design goal*, not current capability. The Q8 result cannot be assumed bit-identical to exact F32.

## Exact current implementation / correctness boundary

- `ggml/include/ggml-backend.h`: 1250 adds `ggml_backend_comm_allreduce_tensor_fused_add_t`.
- `ggml/src/ggml-backend-meta.cpp`: 1250 resolves optional `comm_allreduce_fused_add`; matches the next subgraph's direct or reshape-only F32 ADD with a single-use reduced operand, same shape, MIRRORED residual/output; resolves residual/output for each rank; on success sets `skip_node` and temporarily clears the ADD's `GGML_TENSOR_FLAG_COMPUTE` during `ggml_backend_graph_compute_async`. On unsupported/failure it calls the ordinary AllReduce. **These are source-level gates, not proof of runtime graph replay or atomic fallback.**
- `ggml/src/ggml-cuda/allreduce.cu`: 1250's `ggml_cuda_ar_allreduce_fused_add` accepts only two-rank Q8_0 and calls `ggml_cuda_ar_allreduce_copy_q8_outer` or rejected-provider `ggml_cuda_ar_allreduce_p2p_q8_outer`. `1272` owns `ggml_cuda_ar_launch_finish` and `ggml_cuda_ar_q8_0_add_residual_kernel`, not 1250.
- `getenv("GGML_CUDA_AR_FUSED_RESIDUAL") != nullptr` treats `GGML_CUDA_AR_FUSED_RESIDUAL=0` and `off` as **enabled**, subject to the other gates. Require explicit `1` admission or documented nonempty-only semantics before any experimental deployment.
- The patch mechanics test `tools/tests/patch/test_1250_nro01_allreduce_q8_wire.py` applies 1252→1272→1250 and asserts code presence/idempotence. It does **not** exercise real graph suppression, provider failure after enqueue, aliasing, or numerical equivalence. The old TESTING instructions to enable P2P on Brutus are withdrawn.

**Failure/lifetime requirements for any successor:** prove all-rank matcher identity and rank mapping; one-consumer and reshape/view alias safety; residual/output non-overlap or explicitly supported in-place semantics; valid output before suppressing ADD; graph capture/update/replay with compute flags restored on all return paths; same-process multi-request/multi-ubatch; no fallback after a partial fused write/enqueue unless completion and rollback are proven. An ordinary provider returning `false` after enqueue is not automatically a transaction. Fail closed, preserving the native ADD, on any uncertainty.

## Ownership / consolidation

- **PNRO02** owns only the logical `AR(partial)+mirrored_residual` epilogue and its semantic/admission decision; the old PGC06 CLI concept is subsumed and must not become a second selector.
- **PGC09/PGC12** own validated exact-F32 host/RCCL routing and phase/provider evidence; **QFP01/1291** owns the current Flash-Next CPU-root collective. Prefer its existing result-write seam for a proven eligible residual, rather than resurrecting 1250/1252.
- **PGC16** owns a *different* rewrite, `AR(a)+AR(b) → AR(a+b)` for compatible PARTIAL branches. A MIRRORED residual is not another PARTIAL: moving it inside reduction would multiply it by rank count. No shared generic graph-rewrite framework is justified.
- **PNRO03/PNRO18/1252** remain terminal on current hardware; **1272** owns Q8 codec and shared finish. Do not change their state, code, or active hardware lanes as part of this item.

## Bounded implementation and validation gate

1. **Read-only topology census** on a supported current production model: instrument existing Meta/CPU-root collective boundary and graph consumer counts to find actual `AR(partial) → optional no-reorder reshape → ADD(mirrored)` instances. Record rank set, tensor type/shape, source/destination aliases, provider, ADD launch count and exclusive ADD critical-path time; do not treat a source matcher or process-once marker as activation. Capture pp512/2048 and tg128/MTP verification separately on dual gfx1100 and, only if present, gfx1201. Never assume gfx1030 or N=3 supports the two-rank epilogue.
2. **Cheap discriminator:** a CPU/host graph fixture for direct ADD, 1/2 reshape, reversed operands, extra consumer, non-mirrored residual, shape mismatch, in-place alias, provider-unavailable, partial-enqueue failure, repeat request, graph replay and `env=0`. Require unchanged output and ordinary ADD for every rejected case. Confirm rank-group equality and exactly one ADD removed only on accepted cases.
3. **Materiality before code:** measure exclusive ADD wall time (not collective wall). Even eliminating ADD completely needs `f_ADD >= 1-1/1.03 = 2.913%` of end-to-end time to reach a 3% speedup; if below, close `NO_CRITICAL_PATH`. If no eligible production topology, close `NO_MATCH`. The historical 1–2.5% estimate is a hypothesis, not BigCherry evidence.
4. **Only if both gates pass:** prototype one **default-off, exact-F32, host/CPU-root provider-owned finish** using existing result-write and Meta matcher primitives. No 1252 dependency, no Q8 wire, no second transport, scheduler, allocator, registry or CLI. Specify stream/event lifetime and success-before-skip protocol. Keep unmodified provider/ADD as fallback.
5. **Correctness then performance:** same-process repeated requests, multi-ubatch, long-context, graph on/off, full-vocab logits/KLD/greedy parity, MTP acceptance, expected-vs-observed collective/ADD counts, and memory/transfer accounting. Then >=4 independent sessions, >=10 paired ABBA rounds per architecture and pp/decode control. Promotion: CI95-low >=3% end-to-end on an eligible workload, no >1% control regression, and all correctness gates. Otherwise close `NUMERIC_FAIL`, `UNSAFE_LIFETIME` or `NO_GAIN`.

## Upstream / other engines (checked 2026-10-10)

- [llama.cpp #27825](https://github.com/ggml-org/llama.cpp/pull/27825) merged HIP internal AllReduce support; it does not qualify this residual epilogue or a rejected P2P provider.
- [ROCm/AITER #6270](https://github.com/ROCm/aiter/pull/6270) (open 2026-10-10) adds a residual-identical-on-all-ranks fused AR writeback and separate 1-/2-stage admission. Its actual changed `custom_all_reduce.cuh` and Python communicator provide **semantic and fallback reference**, not a drop-in no-P2P implementation.
- [vLLM fusion design](https://github.com/vllm-project/vllm/blob/main/docs/design/fusions.md) limits its FlashInfer AR+residual+RMSNorm fusion to supported NVIDIA SM90/SM100 and warns about TP+DP/PP combinations. [vLLM #59483](https://github.com/vllm-project/vllm/pull/59483) (open) explores ROCm AITER AR+RMSNorm+MXFP4 quant for tuned small-M; not directly transferable to BigCherry's GGUF and heterogeneous PCIe topology.

## Evidence and history

First-party measured **provider** evidence, not PNRO02: validated dual-XTX 0840 exact-F32 adaptive `tg128 +3.87..3.98%`, `pp512 +0.02..0.04%`, bit-identical; CPU-root 1291 Flash-Next adoption ~5–6% decode (BPB01). No measured PNRO02 speedup or current-pin graph-activation receipt.

2026-10-10 audit: 12 source-static checks and 12 disposable host-admission cases passed; Q8-vs-F32 synthetic 32-element fixture differed at 32/32 values (max abs ~0.0207682). **Not** GPU execution, build, repository pytest or model inference. PNRO02 last independent change 2026-10-03 15:22 UTC, outside 12-hour window; 2026-10-08 BPB01 reviewed 1250 without promoting it. Protected Radiance/Flash-Next 1330/1334/1347/1355/1356/1357/1358/QFP43 work was not modified.

Historical sequence: 2026-09-24 1251 was a kernel-only scaffold; 2026-09-25 1250 added Meta matcher/hook and removed 1251; 2026-10-04 PGC06 ownership consolidated here; 2026-10-08 1252 rejected and BPB01 recorded infeasible 1250 composition. Supersedes NRO02.
