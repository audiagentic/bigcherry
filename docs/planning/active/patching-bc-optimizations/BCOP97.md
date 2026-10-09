---
id: BCOP97
order: 97
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-09T18:10:00+00:00'
created-by: agent
priority: P2
work: S
---

# PGC10 / 1276: gate three-rank root3 against deployed CPU-root

## Discovery / disposition

1276 composes 0840+1244 for **explicit** N=3 adaptive routing and configurable root rank, but 0840's automatic default remains two gfx1100 only. Root3 is F32-only, has its own copy-threshold gate and is `untested`; default 96 KiB logical switch would send an 80 KiB prompt tail to root3. Validated 1291/QFP01 already handles <=64 KiB contiguous F32 N=3 through a per-comm CPU worker and RCCL above; its measured Flash-Next +6.4% plain/+5.1% MTP decode is not a matched root3 comparison. Historical 1244 root3 +2.0% tg512/+3.4% tg2048 and -68% prefill (without crossover) do not establish advantage over 1291. No new GPU measurements.

## Ownership / recent work

PGC10 owns the remaining three-provider decision; GP11/1244 owns root3 protocol, QFP01/1291 owns deployed CPU-root, PGC09 dual-XTX adaptive, PGC12 phase routing, 1225 RCCL admission. Latest independent PGC10/root3 work found 2026-09-30, 1291 work 2026-10-04; no matching independent 12-hour patch/plan/PR/queued lane. Recent BCOP94/96 address Meta views/IQ MMVQ, not N=3 AllReduce. Protected 12-hour Flash-Next accuracy (1334/1347/1350), QFP35/36/41, 1357/1358, Radiance and engine/lab work was not edited.

## Bounded action / terminal

First use existing 1276 composition tests plus a host admission/alias fixture; then require actual three-card CPU-root bottleneck evidence before a **single isolated** RCCL/CPU-root/root3 A/B. Cover 10/20/64/80/120 KiB and >=1 MiB, rank roots 0/1/2, repeated graph replay, inactive shards, MTP and long context; verify actual provider completion, rank/byte work, correctness and no hang. Four sessions x >=10 ABBA pairs; CI95-low >=3% E2E decode over CPU-root, no prefill/control regression beyond PGC10 limits. If bottleneck absent or gates fail, close 1276/PGC10 and retain validated 1291+RCCL. No new provider registry, calibration cache or experiment queue.

## Evidence / sources

PGC10; QFP01; GP11; 0840, 0860, 1244, 1276, 1291 patch.py and metadata; `tools/tests/patch/test_1276_ar_adaptive_nway.py`; [llama.cpp #27825](https://github.com/ggml-org/llama.cpp/pull/27825), [#29793](https://github.com/ggml-org/llama.cpp/pull/29793); [vLLM QuickReduce](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/quick_all_reduce.py); [SGLang #31117](https://github.com/sgl-project/sglang/issues/31117). 22 source/host-model conditions checked (initial static assertion corrected and rerun); no repository pytest, compilation, hardware run or benchmark.
