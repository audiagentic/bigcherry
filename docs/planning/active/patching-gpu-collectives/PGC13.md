---
id: PGC13
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-10-01T07:56:23.005200+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Per-topology AllReduce calibration: offline discriminator, no runtime policy cache

## Audit disposition (2026-10-09)

**Narrow PGC13 to one bounded offline comparison against the validated 96 KiB dual-gfx1100 exact-F32 policy.** The previous startup auto-calibration, phase/size table, per-topology cache and per-card `-ts` weight controller are **not implementation-ready**. Do not add a new scheduler, provider registry, allocator, calibration CLI, cache schema, startup service or runtime flag. No BigCherry calibrated speedup has been measured.

Ownership: PGC13 owns only the offline threshold decision; PGC09 owns the production adaptive provider and configuration; PGC12 owns phase provenance and provider-completion/transition traces; PGC11 owns wire accuracy; PGC14 owns RCCL; QFP01/1291 owns validated N=3 CPU-root; PGC10/1276 owns experimental root3; PGC15/16 own prefill overlap/count reduction. Model placement stays with existing split owners. Do not edit or queue protected QFP35/36/41, 1355/1357/1358, Flash-Next, Radiance, MTP or engine work.

## Exact implementation and data flow

1. `engines/llamacpp/patches/0860_allreduce_provider_cli/patch.py` owns `ADAPTIVE_SWITCH_BYTES_DEFAULT = 96 << 10` and process-wide `ggml_backend_cuda_comm_config` applied after CLI parsing. It has a single provider/wire/switch value, **not** a policy object or per-phase lookup.
2. `engines/llamacpp/patches/0840_hybrid_allreduce_dispatch/patch.py::ggml_backend_cuda_comm_init_hybrid` snapshots `switch_bytes` into `comm_ctx->adaptive_switch_bytes`. `ggml_backend_cuda_comm_try_allreduce_hybrid` uses `ggml_nbytes(tensors[0]) < switch_bytes` to prefer exact-F32 host for small reductions and RCCL for large. Auto admission requires two distinct physical gfx1100 devices with RCCL eligibility. It has **no phase hint**. If an attempted RCCL reduction fails, it returns false rather than automatically retrying host after possible partial execution; do not describe this as symmetric fallback.
3. `engines/llamacpp/patches/1277_ar_size_trace/patch.py` records only bytes, `ne0/ne1`, dtype and the *predicted* route from `prefer_internal`; it has no provider-completion result, phase, graph/ubatch, transition timing or rank identity. Its static non-atomic trace counter is unsuitable for concurrent evidence. Use it only as a PGC12 lab instrumentation anchor, not a runtime policy source.
4. `engines/llamacpp/patches/1291_ar_cpu_root/patch.py` has a separate N=3 CPU-root worker and 64 KiB default small-message limit. Its measured gains cannot be compared directly with N=2 0840 or root3. `tools/lab/rccl/ar-latency.hip` measures RCCL calls in isolation/stream-ordered chains, **not** in-model first-switch or graph replay. Reuse `tools/lab/native-vs-patched/server-ab-adaptive-switch*.json`, `tools/lab/ar-accuracy` and existing rocprof/ABBA receipts; do not add another benchmark harness.
5. PGC12's pp2048 profile reports ~0.93 s compute per device versus ~2.4 s AllReduce service; this is not proof that an isolated latency crossover improves the critical path. An earlier 1 MiB plain pp1024 lane regressed ~4.2%, with a four-token prompt tail triggering 128 host reductions of 80 KiB. The ~50 ms extra cost is not yet attributed to first-switch, launch, staging or synchronization.

### Evidence classes: do not pool different wire/topology contexts

| First-party evidence | Interpretation |
| --- | --- |
| Validated `t-0840e-gfx1100-s1..s4` exact-F32 96 KiB: tg128 +3.87%..+3.98%; pp512 +0.02%..+0.04%; bit-identical | Current production control, **not** an auto-calibration result |
| Plain pp1024 at 1 MiB ~-4.2% vs RCCL; earlier 64 KiB ~-0.07% | Prompt-tail problem, mechanism unresolved |
| Historical BF16-wire MTP tg512 +7.98% at 1 MiB; later exact-F32 MTP -1.4% at 1 MiB, +0.3% at 96 KiB | Distinct wire/configuration; never use BF16 gains to select an F32 phase policy |
| 1291 N=3 CPU-root +6.4% plain and +5.1% MTP decode | Different provider and topology; no matched 0840 comparison |

A disposable **synthetic** host fixture demonstrated that the fastest isolated provider can lose over a sequence when switching incurs cost. This is a mathematical discriminator, **not a measured first-switch penalty**. A separate synthetic receipt-admission fixture rejected mismatched rank, UUID order, driver, RCCL linkage, wire, phase, correctness and control identity.

## Gate 0: cheapest discriminator; no hardware queue

PGC12 must first extend lab-only 1277 instrumentation to correlate `{graph_uid,ubatch,phase,physical_rank_set,dtype,wire,nbytes,provider_preference,provider_completed,first_transition_us,rccl_linked}` with actual provider return and existing trace/rocprof evidence. Do not claim phase from shape (`ne1`) or a preference log. Collect trace outside timed runs. Require a fixed production binary, model GGUF, context/MTP depth, ROCm/kernel, UUID/order, PCIe width, RCCL linkage and graph mode. Missing metadata => `INELIGIBLE`. Preserve inactive-shard zeroing and provider graph lifetime.

First prove a reproducible exact-F32 96 KiB bottleneck in **one** existing dual-gfx1100 production workload, and attribute its end-to-end critical-path share (not the sum of overlapping per-rank kernel times). If absent or optimistic E2E gain ceiling <3%, **close PGC13 without implementation**. Do not run speculative link sweeps 4 KiB..64 MiB or per-card MMVQ/MMQ `-ts` calibrations. Arrival skew, split legality and graph ownership make isolated card timings insufficient for a placement policy.

## Gate 1: only after gate 0 passes

Compare exactly three **existing** scalar settings `--allreduce-switch-bytes 65536/98304/131072` (96 KiB baseline) using same dual XTX, exact-F32 wire, same rank order and model. Four independent sessions, >=10 paired ABBA rounds/session, pp256/512/1024/4096, tg128/512, MTP verify widths 2-5, repeated requests and long context. Record provider completion/first transition in separate diagnostic runs; measure timed controls without diagnostic overhead. Require full-vocab logits/greedy/KLD, MTP acceptance within established exact/exact noise, graph capture/replay, inactive-shard correctness, no missing work or hangs, and <=0.5% prefill regression. Promote only CI95-low >=3% **E2E** over validated 96 KiB on the identified bottleneck, with no negative-control regression. Failure closes PGC13; success changes only an explicit **launch-profile threshold** under PGC09, not runtime calibration or cache.

Only if two real phases at the *same size and wire* demonstrably need different providers may PGC12 separately design a default-off phase hint with per-call lifetime and graph replay safety. PGC13 cannot independently implement it.

## Upstream and external mechanism decisions (inspected 2026-10-09)

- [llama.cpp #27825](https://github.com/ggml-org/llama.cpp/pull/27825) merged HIP internal AllReduce; [#29793](https://github.com/ggml-org/llama.cpp/pull/29793) fixed inactive-shard zeroing via FILL. These are the stock correctness/transport baseline, not a phase-aware policy cache.
- [vLLM QuickReduce](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/quick_all_reduce.py) gates by supported world size (2/4/8), connectivity, regime and message size. Adopt its **fail-closed admission principle**, not its GPU implementation for no-P2P three-card RDNA.
- [SGLang PCIe-IPC](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/device_communicators/pcie_ipc_ar.py) bounds workspace to decode shapes, tunes only admitted shapes outside graph capture, persists covered tactics and reports zero-shape tune failure. It disables this path for deterministic inference because per-shape tactics can change reduction order. Its FlashInfer BF16 and external SM120/8-rank gains do **not** transfer to exact-F32 HIP RDNA. Adopt its warmup/coverage/fallback discipline, not its backend or cache.

## Terminal disposition and history

Reject/close if missing actual provider/phase evidence, absent bottleneck, ceiling <3%, mismatched binary/topology/wire, failed correctness, CI95-low below threshold or prefill regression. Do not introduce a second experiment or auto-calibration layer. Keep PGC09's validated 96 KiB default until an independently qualified replacement.

2026-10-01: broad auto-calibration/cache proposal (superseded).
2026-10-09: source-level audit narrowed to offline scalar-control gate; BCOP98.
