# Testing — NRO05 GDN MTP prefix/tail

Target: Qwen3.8-27B-Q8_0, dual 7900 XTX/gfx1100, `-sm tensor`, MTP `n_max=4`. Subject includes required patch 1253.

## Activation
Run with `BIGCHERRY_PATCH_TRACE=1`; require subject hit and control miss for `BIGCHERRY_PATCH_HIT patch=1254_nro05 path=gdn_mtp_prefix_bf16`. Also run subject with `GGML_CUDA_GDN_CHUNKED=0` and require the optimized route not to fire.

## Correctness
Use the contract producer and compare pre-sampling, full-vocabulary MTP logprobs control vs subject; where `validation.toml` requires backend-reference parity, require max absolute logprob difference <= `5e-4`. Preserve greedy token parity. Separately exercise K=2/3/4/5/8, threshold boundaries, all snapshot slots, final recurrent state, continuation, and chunked-launch rejection/fallback.

## Work equivalence
Record drafted/accepted counts and acceptance ratio for every arm. MTP draft acceptance must be identical/parity-compatible before throughput is attributable; changed speculative work voids throughput attribution.

## Performance
Use the matching order-balanced configs under `tools/lab/native-vs-patched/server-ab-*.json`; alternate control/subject order, retain per-pair raw records, and report prefill and decode separately. Existing `runs/nro05-ab2/RESULT.md` measured 8 balanced pairs: pp4096 +2.12%, decode flat, identical acceptance. Treat this only as measured benchmark evidence; do not claim validation/sign-off until the declared correctness and controls pass.
