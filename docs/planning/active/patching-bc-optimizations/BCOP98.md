---
id: BCOP98
order: 98
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-09T19:12:00+00:00'
created-by: agent
priority: P2
work: S
---

# PGC13: gate topology auto-calibration before any runtime policy cache

## Discovery / disposition

0860 exposes a 96 KiB scalar switch, 0840 snapshots it per comm context, and no per-call phase or policy cache exists. Lab-only 1277 logs predicted rather than completed provider and omits phase/first-switch attribution. A synthetic host fixture shows isolated per-call wins can lose over a switched sequence; **no new GPU measurement or actual first-switch cause** was established. Historical 1 MiB BF16 MTP and current exact-F32 96 KiB results cannot be pooled. Defer/retire the proposed runtime auto-calibrator, topology cache and `-ts` solver; retain a bounded offline scalar-control decision only.

## Owners / exclusions

PGC13 offline gate; PGC09 production threshold; PGC12 actual phase/provider trace; PGC11 wire; PGC14 RCCL; QFP01/1291 N=3 CPU-root; PGC10/1276 root3. PGC13/09/12 last independent updates October 1; no independent 12-hour implementation, plan, PR, hardware lane or queue. Protected QFP35/1355, QFP36/1357, QFP41/1358, Radiance, Flash-Next accuracy, MTP and engine/lab work untouched. Prior BCOP88-97 did not own topology auto-calibration.

## Bounded next action / terminal gate

PGC12 first proves real provider completion, graph/ubatch/phase and first-switch cost on an RCCL-linked exact-F32 dual-XTX workload. No 96 KiB bottleneck or theoretical E2E ceiling <3% => close PGC13. Otherwise one 64/96/128 KiB existing-CLI A/B, four sessions x >=10 paired rounds, prefill/decode/MTP, graph replay, full-vocab and no missing work; require CI95-low >=3% E2E over 96 KiB, <=0.5% prefill regression and preserved correctness. Success changes only a static PGC09 launch profile; failure closes. No new dispatcher, cache, calibration CLI or hardware queue.

## References / validation

PGC09/12/13; 0840/0860/1277/1291 patch.py; `tools/lab/rccl/ar-latency.hip`; [llama.cpp #27825](https://github.com/ggml-org/llama.cpp/pull/27825), [#29793](https://github.com/ggml-org/llama.cpp/pull/29793); [vLLM QuickReduce](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/quick_all_reduce.py); [SGLang PCIe-IPC](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/device_communicators/pcie_ipc_ar.py). 12 source-static checks (one assertion typo corrected), 8 synthetic sequence and 11 synthetic receipt-admission assertions passed. No repository pytest, compilation or GPU benchmarking.
