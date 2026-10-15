---
id: PRBE115
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-29T23:14:07.176425+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Q8_0 MTP verify widths 2–5: exact Q8_1 admission, not raw-F32 widening

## 2026-10-14 b11552 implementation audit (BCOP200)

**Disposition: pending, receipt-first.** Retire the generic "hoisted dequant" suggestion and any repeat of rejected 1301. Stock Q8_1 quantization and integer dot are not numerically equivalent to 1241's raw-F32 FMA; a width2–5 verify kernel cannot inherit width1 performance or claim unchanged MTP acceptance. Do not queue hardware while Radiance/helper work is active.

### Implementation and ownership

- `ggml/src/ggml-cuda/mmvq.cu::ggml_cuda_mul_mat_vec_q`: allocate `src1_q8_1` via `ctx.pool()`, invoke `quantize_row_q8_1_cuda` on `ctx.stream()`, then `mul_mat_vec_q_switch_type`. The `mul_mat_vec_q` kernel loops over width `j` and reuses quantized activations across output rows; `vecdotq.cuh::vec_dot_q8_0_q8_1` uses packed integer dot/scale. Widths 2–5 already have compile-time cases. Preserve pool lifetime, stream order, graph capture, GLU and forced-autotune behavior.
- `0600_mmvq_geometry/patch.py` adds explicit nwarps/rows-per-block template parameters; `1241_rd33_mmvq_q8_0_f32_decode` **requires 0600**. Its gfx1100, dense Q8_0, width1, `!forced.requested()` bypass uses `vec_dot_q8_0_f32` and FMA on raw input. A pristine-b11552 template mismatch is **not** proof of failure after 0600 composition.
- Rejected `1301_prbe115_q8_f32_mtp_widths/patch.py` widens only that gate/launcher; it does not reproduce stock Q8_1 arithmetic. Raw F32 payload is 128 bytes/32 values versus 36 bytes/Q8_1 block (3.56x), **not measured DRAM traffic**. Repeated FMA/dequant, occupancy and cache pressure are hypotheses, not established causes.
- PRBE113/BCOP180 owns the **stock width2–5 per-dispatch cost receipt**; PRBE115 owns only the conditional exact-Q8_1 candidate. 1241 owns width1, 0600 geometry, PRBE12/1206 reshape/add fusion (separate safety gate), QFP48 MMQ *prefill* fusion. No new dispatcher, allocator, cache, scheduler, telemetry or duplicate GPU queue.

### Existing measured evidence (not new/current-pin results)

1241 width1 dual-gfx1100 Qwen3.8-27B Q8_0 tg128 +4.56%..+4.67% raw in four historical sessions; Q6_K control +~1.4%, control-adjusted ~+3.2%; backend CPU-reference 50/50. QFP21's 2026-10-05 MTP4 width2/3 1301 tests at 10K/32K were neutral. MAXCOLS5 regressed ~12% at 10K (~73 to 64.7–64.8 tok/s), ~6% at 32K (~72 to 67.8–67.9). Earlier Flash-Next acceptance drift was not a fixed-draft causal test. Patch 1245 width6 lost ~9.4% in a separate single run (PRBE113). No new measured gain.

### Gate 0 — cheapest discriminator before hardware

Read **terminal** PRBE113 receipts for exact Qwen3.8-27B Q8_0 digest, b11552/build/patch set, ROCm, two gfx1100 UUIDs and rank order, fixed MTP4 draft source, 10K/32K/80K+ context, steady decode, stream/graph/ubatch. Require per-width 2–5 calls, quantize Q8_1 launches/bytes, MMVQ, GLU, GDN, AllReduce, PCIe/CPU gaps, critical-path wall share, work/token and acceptance counts. Summed concurrent GPU busy time is not exposed E2E latency. If absent, PRBE113 alone may queue one stock capture **after** active hardware lanes terminate. Close `INELIGIBLE_ROUTE` if no matching widths execute.

A 3% throughput target requires removal of at least `1-1/1.03 = 2.9126%` of *serialized baseline latency*. Exact Q8_1 quantization cannot simply disappear; potential saving is only exposed launch/materialization/allocation or duplicated producer work minus fusion overhead. If optimistic exposed removable cost is smaller, close `BELOW_CEILING` without implementation.

### Gate 1 — conditional exact-Q8_1 producer

Only if Gate 0 passes: select **one** existing F32 producer with single MMVQ consumer, known strides and device, and proven stream/graph/buffer lifetime. Reuse its existing owner to emit *byte-identical* stock Q8_1 blocks (scale, rounding, clamp, sum, padding), then use the **existing** MMVQ consumer. Never quantize inside each output-row dot. Pseudocode:

```text
if same_device && single_consumer && exact_shape_stride && Q8_1_bytes_equal
   && producer_before_MMVQ && graph_lifetime_proven && !forced_autotune:
    existing_producer_emits_Q8_1_to_existing_MMVQ()
else:
    stock_quantize_row_q8_1_cuda_then_MMVQ()
```

First compare all Q8_1 bytes against stock for zero/ties/extremes/padding/width2–5 and run a host graph-alias/stream fixture. Reject `NO_EQUIVALENT_PRODUCER` if exact bytes require duplicate per-row work or new global state. Do not bundle 0600 geometry changes to disguise the result.

### Gate 2 — correctness/performance promotion

Require full-vocabulary logits/KLD and greedy identity, fixed-draft MTP acceptance (depth1/4/5), multi-request same process, alternating ubatches, graph replay/rebind, 10K/32K/80K+, correct work/transfer accounting, forced-autotune negative control, peak VRAM and no P2P assumptions. Qualify dual gfx1100; gfx1201 only with independently eligible route; gfx1030 excluded. Four independent sessions per architecture, >=10 balanced paired rounds each; CI95-low >=3% E2E decode, <=1% control regression, zero numerical/acceptance drift. Otherwise `NUMERIC_DRIFT`, `NO_GAIN` or `UNSUPPORTED_ROUTE`; retain stock.

### Upstream and alternative engines

[llama.cpp #25618](https://github.com/ggml-org/llama.cpp/issues/25618), updated 2026-10-12, warns quantized N=1/N>1 speculative greedy output can diverge on Vulkan; not proof of HIP cause. Open [#30021](https://github.com/ggml-org/llama.cpp/pull/30021) tunes GCN **MMQ**, not MMVQ verify. [vLLM typed FP8 activation fusion](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/act_quant_fusion.py) is not GGUF Q8_1. [SGLang GGUF dispatch](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/gguf.py) warns HIP GGUF unsupported. Reference admission only; no transferable gain.

**This audit:** pinned/patch source assertions 12/12; simplified host arithmetic 6/6 (500/500 seeded raw-vs-quant-like dots differ). Host fixture is not HIP-bit-exact. No repository pytest, patch application, HIP build or GPU run.

## Change log

- 2026-10-14 BCOP200: reconciled 0600/1241/1301, retired raw-F32 widening, required stock-equivalent producer and PRBE113 cost receipt.
- 2026-10-08: P2 profile-first triage after 1301 rejection.
- 2026-10-04/05: 1301 Flash-Next and Q8_0 MTP width screens; rejected.

## Ledger-events

- chg_20261003_141341_experimental-decode-kernel-pat_2923
