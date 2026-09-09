# RHA04 required dual-XTX 27B matrix

This is the completed six-permutation, three-arm server-bench capture for the
Qwen3.8-27B Q8_0 model on Brutus dual XTX. The run used the maintained test
bench runner (`run_bench.py` via `bigcherry ab-benchmark`), not llamabench.

- Capture: `run.json`, 18 arm cells (six permutations × replay/stock/native)
- Model: `Qwen3.8-27B-Q8_0.gguf`
- Server: tensor split, `-sm tensor --fit off`, 4096 context, batch 2048,
  ubatch 512, one slot
- Timed environment: `HIP_VISIBLE_DEVICES=0,1`, `ROCR_VISIBLE_DEVICES=0,1`
- Preflight-only diagnostics: `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT`
- Physical identity: RCCL `cudaDev 0 -> 0000:03:00.0`, `cudaDev 1 -> 0000:06:00.0`
- Every cell: return code 0, clean SIGINT shutdown, `execution_evidence_status=verified`

## Per-arm medians (tokens/s)

| arm | pp512 | pp2048 | tg128 | tg512 |
|---|---:|---:|---:|---:|
| stock | 927.72 | 1286.465 | 33.97 | 34.22 |
| native | 928.30 | 1288.885 | 33.74 | 34.04 |
| replay | 930.88 | 1287.295 | 34.10 | 34.32 |

Relative to stock medians, native is +0.06% / +0.19% / -0.68% / -0.53%
and replay is +0.34% / +0.06% / +0.38% / +0.29% for pp512 / pp2048 /
tg128 / tg512 respectively. The stock pp512 sample in pair 5 was 772.35
tokens/s (the raw log is retained); it is a real host-side prompt-throughput
outlier, so the full six-sample medians are reported without silently removing
it. The complete per-cell values and raw logs are in `run.json` and the
`pair-*` directories.

`performance_admitted` remains false by design: source/work-equivalence
provenance and separate replay activation admission are still required. This
capture establishes the dual-device execution gate and provides parity evidence;
it is not a claim that replay is production-approved.
