# 0860 AllReduce provider CLI testing

## Scope
Validate the explicit AllReduce CLI/configuration seam across `llama-server`, `llama-bench`, and `common/arg.cpp`: `--allreduce auto|ccl|host|adaptive|p2p|root3|butterfly` and `--allreduce-wire native|q8`. Legacy `GGML_CUDA_ALLREDUCE` and `GGML_CUDA_AR_WIRE` are removed selectors and must not affect selection.

## Hardware-free
- `PYTHONPATH=tools python -m unittest tools.tests.patch.test_0860_allreduce_provider_cli`
- `PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --focal-overlay 0860_allreduce_provider_cli`
- Exercise both `llama-server` common-argument parsing and `llama-bench` benchmark-local parsing. Reject unknown providers/wires and comma-list values; require startup failure rather than fallback.

## Provider/startup matrix
Hardware: dual gfx1100 (2x Radeon RX 7900 XTX), RCCL-capable build, no P2P, `-sm tensor`. Set `BIGCHERRY_PATCH_TRACE=1`.

Run `llama-server` and `llama-bench` with each supported provider. `auto`, `ccl`, `host`, `adaptive`, and `butterfly` must start when their implementation patch is present and emit `BIGCHERRY_PATCH_HIT patch=0860_allreduce_provider_cli provider=<x> wire=<y>`. `auto` must resolve to `ccl` on Linux and `host` elsewhere. Record resolved provider/wire and command line for every case.

Fail closed: base 0860 must reject `p2p` and `root3` as `not available in this build` until their implementation patches are present. `--allreduce-wire q8` with any non-`p2p` provider must exit non-zero with the unsupported-combination diagnostic. Unknown provider/wire names must exit non-zero at startup with a clear validation error. When the p2p implementation is absent, `--allreduce p2p --allreduce-wire q8` must also fail as unavailable.

Legacy-selector negative test: set `GGML_CUDA_ALLREDUCE` and `GGML_CUDA_AR_WIRE` to values conflicting with the CLI/default, repeat startup, and confirm the marker/resolved provider is unchanged. Repeat with CLI omitted to prove the legacy variables do not alter `auto` resolution.

## Correctness
Model/workload: Qwen3.8-27B-Q8_0, dual gfx1100, `-sm tensor`, MTP `n_max=4`. Compare `ccl`, `host`, and `adaptive` using identical prompts/seeds/settings. Require identical greedy generated output and MTP draft-acceptance parity; reference acceptance is 0.901. Record token streams, accepted/drafted counts, acceptance ratio, provider marker, and any fallback/initialization diagnostics.

## Performance
Use order-balanced `bigcherry ab-benchmark` pairs with provider as the only variable. Compare relevant provider pairs on the same dual-gfx1100 host/build/model and report `pp1024`, `pp4096`, `tg512`, and `tg2048`. Preserve pair order, raw observations, medians/deltas, and Mann-Whitney results; do not combine correctness-divergent runs into performance conclusions.
