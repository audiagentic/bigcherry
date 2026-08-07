---
name: tune-comparison
description: Compare native HIP dispatch against a tuning replay cache using identical llama-bench or server workloads, and verify replay coverage and misses.
---

# Native versus tuned comparison

Use this workflow only after a tune has produced a measurements JSONL file and a replay cache. It is model-family agnostic.

## Required inputs

Set these for the workload under test:

- `MODEL_PATH` — GGUF model path
- `CACHE_PATH` — replay cache exported from the tune
- `DEVICE_ARGS` — identical device/layer/split arguments for both runs
- `BENCH_ARGS` — identical prompt, generation, batch, ubatch, cache, and thread arguments
- A quiet GPU with the same driver, clocks, and power state for both runs

Do not compare runs made with different model files, GPU devices, batch sizes, context settings, or concurrent workloads.

## Step 1 — Run the native baseline

Native dispatch is the before measurement:

```powershell
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
$env:GGML_HIP_DISPATCH_MODE = 'native'

& 'C:\bcw\bin\llama-bench.exe' `
  -m $env:MODEL_PATH `
  -dev ROCm0 -ngl 99 -ctk f16 -ctv f16 -fa auto `
  -b 2048 -ub 512 -t 8 -p 512 -n 512 `
  -r 10 -o json > native.json
```

Replace the example device and benchmark arguments with the exact workload configuration. `llama-bench` does not accept the server-only `-c` option.

On brutus, use the equivalent `~/bc-build-multi/bin/llama-bench` command and save the output to a separate file.

## Step 2 — Run the replay/tuned measurement

Use exactly the same command arguments and only change dispatch mode plus replay diagnostics:

```powershell
$env:GGML_HIP_DISPATCH_MODE = 'replay'
$env:GGML_HIP_DISPATCH_CACHE = $env:CACHE_PATH
$env:GGML_HIP_DISPATCH_COVERAGE = 'replay-coverage.json'
$env:GGML_HIP_DISPATCH_MISS_LOG = 'replay-misses.jsonl'
$env:GGML_HIP_DISPATCH_MISS = 'native-record'

& 'C:\bcw\bin\llama-bench.exe' `
  -m $env:MODEL_PATH `
  -dev ROCm0 -ngl 99 -ctk f16 -ctv f16 -fa auto `
  -b 2048 -ub 512 -t 8 -p 512 -n 512 `
  -r 10 -o json > replay.json
```

The native and replay runs must use the same repetition count. Discard and repeat either run if the GPU was contended or thermally/power-state unstable.

## Step 3 — Compare throughput

The JSON output reports prompt-processing and generation throughput. Compare medians when available; otherwise compare the reported mean and spread:

```powershell
python -c "import json; a=json.load(open('native.json')); b=json.load(open('replay.json')); print('native:', a); print('replay:', b)"
```

For a compact percentage comparison, inspect the benchmark rows and calculate:

```text
improvement_pct = (replay_value - native_value) / native_value * 100
```

Positive is faster. Report prompt and generation throughput separately; a cache can improve one without improving the other.

## Optional per-entry hit diagnostics

Production replay builds do not include hit diagnostics. Build a diagnostic replay binary with:

```powershell
-DGGML_HIP_DISPATCH_REPLAY=ON -DGGML_HIP_REPLAY_DIAGNOSTICS=ON
```

Then set:

```powershell
$env:GGML_HIP_DISPATCH_HIT_LOG = 'replay-hits.jsonl'
```

After graceful shutdown, each JSONL row identifies the dispatch digest, signature digest, replayed candidate, and cold-path hit count. This is the authoritative way to confirm that a tuned winner actually ran. The normal miss log remains separate.

## Step 4 — Verify replay coverage

Inspect `replay-coverage.json`:

- `replay.misses` should be zero for a fully covered benchmark.
- `replay.stale` should be false.
- Replay entries should correspond to the expected cache generation.

Inspect `replay-misses.jsonl` if present. A native fallback can still produce correct output and benchmark numbers, but it is not evidence that the cache handled that dispatch.

A coverage percentage of 100% alone is insufficient: native fallback dispatches may still count as covered. Always check replay misses.

## Step 5 — Optional server-level comparison

For end-to-end latency, run the same server benchmark twice with the same model/server arguments:

1. `GGML_HIP_DISPATCH_MODE=native`
2. `GGML_HIP_DISPATCH_MODE=replay` with `GGML_HIP_DISPATCH_CACHE` set

Send the same request sequence, prompt lengths, generation limits, concurrency, and warm-up requests. Compare time-to-first-token, prompt processing, generation rate, and total latency. Save coverage and miss logs for the replay run.

Do not force-kill a tuning run before its measurements are written. For a patched local server, set `LLAMA_SERVER_ENABLE_SHUTDOWN=1` and send `POST /shutdown`; this runs normal backend cleanup. Comparison runs do not generate tuning records, but replay diagnostics must still be saved before stopping the server.

## Interpretation

- A small change (under roughly 1–2%) may be normal measurement noise.
- A replay result slower than native can indicate cache misses, stale/incompatible entries, contention, or a noisy benchmark.
- A valid tune does not guarantee an improvement for every workload; the cache only contains signatures exercised during tuning.
- Keep the native JSON, replay JSON, coverage JSON, miss log, cache, model identity, revision, GPU identity, and command lines together as the comparison record.
