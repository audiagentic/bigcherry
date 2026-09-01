# Multi-GPU large-model validation: real hardware findings

Real, hardware-confirmed gotchas from RD73's (VA06) `--run-rd73-contract`
qualification on Brutus (dual RX 7900 XTX, gfx1100, `tierL-qwen27b-q8`,
2026-09-01) -- discovered across 9 real hardware attempts. Anyone
building a validation lane that launches `llama-server`/`llama-bench`
against a model large enough to need multi-GPU tensor split should read
this first; each item below cost a real, otherwise-avoidable hardware
cycle to find.

## `llama-bench` cannot reliably run this class of model

For a 27B model split across 2x 24.5GB gfx1100 cards, `llama-bench`
crashed two different ways depending on flags:

- **No `-sm tensor`**: fails to load the model at all -- a real
  `vector::_M_range_check` crash, not merely slower. `-sm tensor` (not
  the default `-sm layer`) is required for a multi-GPU tensor split.
- **With `--fit off`**: `llama-bench` does not register the `--fit`
  flag at all -- `error: invalid parameter for argument: --fit`, a hard
  argument-parse failure. `--fit` is a `llama-server`-only flag (see
  below); passing it to `llama-bench` is always wrong.

Given both failure modes, prefer a real `llama-server` + HTTP-request
harness (or the documented Brutus bench runner, below) over
`llama-bench` for this class of model.

## `llama-server` needs `-sm tensor` AND `--fit off` together

`llama.cpp`'s automatic device-memory-fit feature (`--fit`, default
`on`) is not implemented for `SPLIT_MODE_TENSOR` and aborts with
`llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort`
(`common/fit.cpp`) the moment `-sm tensor` is set. Both flags are
required together on `llama-server`; `--fit off` must **never** be
passed to `llama-bench` (see above -- it doesn't exist there).

## Control and subject servers cannot run concurrently

A paired control/subject validation lane naturally wants to launch both
servers at once for real-time alternating measurement. For a model this
large, that doesn't fit: each server needs ~13GB/GPU under `-sm tensor`
split, and two full copies exceed the 24.5GB/GPU cards -- a real
`cudaMalloc failed: out of memory` abort (confirmed on hardware: the
control server loaded and was listening, then the subject server
aborted mid-load).

Fix used here: launch one fresh server per single measured/warmup
request, alternating control/subject arms in the same order the paired
statistics engine already calls them, rather than holding both open for
the whole lane. This preserves the alternating-order/thermal-drift
discipline real production benchmarking on this project has found
necessary (a non-alternating "all control then all subject" design
previously produced a real, since-corrected measurement artifact on
this exact model/hardware -- see `patches/1233_rd73_stable_graph_cache_key/README.md`'s
"Historical evidence" section), at the cost of a full server/model
reload per single request.

**Consequence for resource/cache-accumulation measurements**: if what
you're measuring depends on cross-request state inside one running
process (e.g. an in-memory cache that grows with repeated-shape
traffic), the per-request-restart pattern above destroys exactly the
behavior you're trying to measure -- a fresh process resets everything.
Use a **separate**, long-lived, single-arm (usually subject-only, so no
concurrent-VRAM conflict) session that drives a real repeated-request
burst, instead of trying to derive that evidence from the paired
alternating lane's logs.

## Prefer the documented Brutus bench runner over raw `llama-bench`

`/mnt/vault/development/llmhosts/llamacpp/bench/run_bench.py`
(`docs/reference/testing/TEST.md`'s "Server benchmark (Brutus bench
runner)" section, `--bench-type server-bench`, endpoint mode) drives an
already-running server via real HTTP requests rather than spawning its
own `llama-bench` process -- sidesteps every issue above. Its two
result-printing code paths use different headers depending on bench
type:

- `bench/lib/bench_orchestrator.py`'s `run_llama_bench()` path prints
  `"Aggregated Results (N test(s)):"`.
- `bench/runners/server_base.py`'s server-bench path (what you get with
  `--bench-type server-bench`) prints `"Extracted Results (N
  config(s)):"` instead.

Both share the same `"  <name>_tps: <value>"` per-config line format
underneath the header, but a parser that only recognizes one header
will silently see a "successful" run with no results to extract. Match
both.

## Summary checklist for a new large multi-GPU model validation lane

- [ ] `-sm tensor` on every `llama-server`/`llama-bench` invocation
- [ ] `--fit off` on `llama-server` invocations only -- never on
      `llama-bench`
- [ ] Never launch two servers needing a large fraction of VRAM each at
      the same time; check the real math (`model_size / gpu_count` vs.
      per-GPU VRAM) before assuming concurrent launch is safe
- [ ] If restarting per-request to solve the above, isolate any
      evidence that depends on cross-request in-process state into its
      own long-lived, non-concurrent session
- [ ] If using the Brutus bench runner, parse both "Aggregated Results"
      and "Extracted Results" headers
