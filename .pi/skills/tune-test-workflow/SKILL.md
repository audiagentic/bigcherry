---
name: tune-test-workflow
description: Run a complete HIP autotune sweep on brutus — from starting a server in tune mode through loading measurements into SQLite, running reports, and exporting the replay cache. Use when the user wants to measure which kernel variants are fastest for their workload.
---

# Tune test end-to-end workflow

This skill walks through running a HIP autotune sweep, loading results, and exporting them as a production-ready replay cache.

## Prerequisites

- SSH access to brutus (`audumla@10.10.100.10`)
- Pre-built multi-arch binary at `~/bc-build-multi/bin/llama-server` (already configured for gfx1100, gfx1201, gfx1030 with `GGML_HIP_AUTOTUNE=ON`)
- For Windows local tuning: `C:\bcw\bin\llama-server.exe` and, for the easiest graceful-exit workflow, `C:\bcw\bin\llama-bench.exe`. If `llama-bench.exe` is missing, build it with `cmake --build C:\bcw --target llama-bench -j 8`.
- An inventory JSON for `workload-max` builds (e.g., `artifacts/mtp-inventory.json`)

## Model path and workload configuration

Set `MODEL_PATH`, `DB_PREFIX`, `PORT`, and the server/benchmark arguments for the model and workload being measured. The tuning mechanism is model-family agnostic; only the operations actually exercised by the selected workload produce signatures.

## Step 0 — Find a free GPU on brutus

```bash
rocm-smi --showuse
```

Device indices: **0,1 = gfx1100 XTX (tensor-split pair), 2 = gfx1201, 3 = gfx1030 (RDNA2)**. A sweep needs ~568 MiB free VRAM. For the 27B MTP model, you need both XTXs (devices 0+1).

## Step 1 — Tune the workload

### Server configuration

```bash
MODEL_PATH=/path/to/model.gguf
DB_PREFIX=/tmp/tune
GGML_HIP_DISPATCH_MODE=tune \
GGML_HIP_DISPATCH_DB="$DB_PREFIX" \
GGML_HIP_TUNE_SCREEN_SAMPLES=3 \
GGML_HIP_TUNE_FINAL_SAMPLES=15 \
GGML_CUDA_DISABLE_GRAPHS=1 \
~/bc-build-multi/bin/llama-server -m "$MODEL_PATH" \
  --host 127.0.0.1 --port 42099 \
  -dev ROCm0 -ngl 99 -c 8192 -ctk f16 -ctv f16 -fa auto -b 2048 -ub 512 -t 8 &
```

Adjust device, context, batch, tensor-split, speculative-decoding, and other arguments to match the production workload.

### Quick test (single GPU)

Use the same pattern with a smaller context/batch if the model or GPU requires it. Keep the model path and all workload parameters explicit.

The server writes `/tmp/t.jsonl.measurements.jsonl` with per-candidate timing data and canonical signature shape metadata (`canonical`, including `ne0`/`ned` dimensions and types). The SQLite loader preserves this in `signature.canonical_json` and links every measurement and winner through `signature_id`.

**Tuning knobs**:

- `SCREEN_SAMPLES=3, FINAL_SAMPLES=15` — standard settings (~10 min/sweep)
- `SCREEN_SAMPLES=1, FINAL_SAMPLES=1` — quick check (correctness only, no performance conclusions)
- `GGML_CUDA_DISABLE_GRAPHS=1` — required; tuning is skipped under graph capture (RV05)

**Never draw a performance conclusion from 1/1 runs.** They can't separate a 1% difference from noise.

## Step 2 — Drive the server with a benchmark

```bash
python3 /mnt/vault/development/llmhosts/llamacpp/bench/run_bench.py \
  --bench-type server-bench --server-url http://127.0.0.1:42099 \
  --model "$MODEL_PATH" \
  --timeout 300 --upload-dry-run
```

## Step 3 — Stop the server

```bash
kill %1  # or: fuser -k 42099/tcp
```

## Step 4 — Load into SQLite for analysis

```bash
cd /mnt/vault/development/llmhosts/bigcherry
python3 -m bigcherry inventory tuning /tmp/t.jsonl.measurements.jsonl \
    --database /tmp/tune.sqlite \
    --manifest artifacts/<revision>/hip-autotune-manifest.json
```

The `--manifest` flag populates full candidate data; without it, you get minimal candidate metadata. For thorough analysis, always include the manifest. Find manifests at `artifacts/<rev>/hip-autotune-manifest.json`.

## Step 5 — Run reports

```bash
# Aggregate statistics (how many won, improvement distribution)
python3 -m bigcherry report summary --database /tmp/tune.sqlite

# Per-signature detail tables
python3 -m bigcherry report signatures --database /tmp/tune.sqlite

# Cross-family breakdown for a specific dispatch (use hex digest from another report)
python3 -m bigcherry report families --database /tmp/tune.sqlite \
    --dispatch abc123...

# Hot signatures by call count (needs observation table merged in first)
python3 -m bigcherry inventory record /tmp/rec.jsonl --database /tmp/tune.sqlite
python3 -m bigcherry report hot --database /tmp/tune.sqlite
```

## Step 6 — Export the replay cache

Build a binary cache from the measurements:

```bash
cd /mnt/vault/development/llmhosts/bigcherry
python3 -m bigcherry.replay_cache /tmp/t.jsonl.measurements.jsonl \
    --manifest artifacts/<revision>/hip-autotune-manifest.json \
    --output dispatch.cache
```

This produces a `dispatch.cache` file that can be loaded by a replay-mode build. The cache is binary (not JSONL) and mirrors `hip-autotune-replay.cpp` byte for byte.

## Step 7 — Validate with replay

Build a replay-only binary (mutually exclusive with `GGML_HIP_AUTOTUNE`). Use the existing multi-arch build or configure a new one:

```bash
cmake -S $BC/vendor/llama.cpp -B ~/bc-build-replay -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGGML_HIP_RCCL=ON \
  -DGGML_HIP_DISPATCH_REPLAY=ON \
  -DGGML_HIP_AUTOTUNE_VARIANT_SET=workload-max \
  -DAMDGPU_TARGETS="gfx1100;gfx1201;gfx1030" \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build ~/bc-build-replay --target llama-server -j
```

Run with the cache:

```bash
GGML_HIP_DISPATCH_MODE=replay \
GGML_HIP_DISPATCH_CACHE=dispatch.cache \
GGML_HIP_DISPATCH_COVERAGE=cov.json \
GGML_HIP_DISPATCH_MISS_LOG=miss.jsonl \
GGML_HIP_DISPATCH_MISS=native-record \
~/bc-build-replay/bin/llama-server ...same args as tune...
```

Check `cov.json` for coverage and miss counts. The `replay` section shows:

```json
"total_dispatched": 16082,
"replay": { "entries": 1155, "misses": 0, "stale": false }
```

Coverage 100% doesn't mean the cache was used — a miss to native is still a dispatch. Check `misses` alongside coverage.

## Step 8 — Optional: generate a slim build

For production, filter the catalog to only variants that won:

```bash
cd /mnt/vault/development/llmhosts/bigcherry
python3 -m bigcherry generate --variant-set replay-slim \
    --inventory artifacts/my-inventory.json \
    --winners /tmp/t.jsonl.measurements.jsonl

# Re-export cache against slim manifest
python3 -m bigcherry.replay_cache /tmp/t.jsonl.measurements.jsonl \
    --manifest artifacts/<revision>/hip-autotune-manifest.json \
    --output dispatch-slim.cache
```

**Order matters**: generate slim first, then export the cache. The CMake build requires the registry to be in the tree already.

## Windows local quick tune (any model)

Use `llama-bench` when the goal is kernel tuning rather than reproducing a long-lived server workload. It exits normally, allowing the HIP tuner to flush its measurements. The local Windows paths used by this project are:

```powershell
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
$env:GGML_HIP_DISPATCH_MODE = 'tune'
$model = 'J:\path\to\model.gguf'
$db = 'J:/development/llmhosts/bigcherry/tune-model'
$env:GGML_HIP_DISPATCH_DB = $db
$env:GGML_HIP_TUNE_SCREEN_SAMPLES = '3'
$env:GGML_HIP_TUNE_FINAL_SAMPLES = '15'
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
& 'C:\bcw\bin\llama-bench.exe' -m $model `
  -dev ROCm0 -ngl 99 -ctk f16 -ctv f16 -fa auto -b 2048 -ub 512 -t 8 -p 512 -n 512
```

`llama-bench` does **not** accept the server-only `-c` context-size option. The output is `${db}.measurements.jsonl`; verify that it exists and contains measurement rows before calling the run valid. Load and export it with:

```powershell
$env:PYTHONPATH = 'tools'
python -m bigcherry inventory tuning "${db}.measurements.jsonl" `
  --database tune.sqlite `
  --manifest artifacts/<revision>/hip-autotune-manifest.json
python -m bigcherry report summary --database tune.sqlite --json
python -m bigcherry.replay_cache "${db}.measurements.jsonl" `
  --manifest artifacts/<revision>/hip-autotune-manifest.json `
  --output dispatch.cache
```

For server tuning, drive `/completion` or `/v1/chat/completions`, then stop the server gracefully. The patched server supports an opt-in local shutdown endpoint: set `LLAMA_SERVER_ENABLE_SHUTDOWN=1` before launch, then send `POST /shutdown`; wait for the process to exit and check for `<DB-prefix>.measurements.jsonl`. Do not force-kill it before checking the file; a normal request/latency result without tuning records is not a valid tune. If using an unpatched build, use `llama-bench` instead.

## Offline correctness check (no server needed)

If you just want to verify the build works before a full tune:

```bash
cd ~/bc-build-multi/bin
HIP_VISIBLE_DEVICES=3 GGML_HIP_DISPATCH_MODE=native ./test-backend-ops test -o MUL_MAT
HIP_VISIBLE_DEVICES=3 GGML_HIP_DISPATCH_MODE=tune \
  GGML_HIP_TUNE_SCREEN_SAMPLES=1 GGML_HIP_TUNE_FINAL_SAMPLES=1 \
  ./test-backend-ops test -o MUL_MAT
```

Expect ~1155 signatures tuned and `2/2 backends passed`. Always run `native` as well — it's the baseline that tells you whether a failure is yours or the hardware's.

## Gotchas

1. **Check `canary_pct` before believing narrow margins.** A 1% difference might be noise. The canary measures the same kernel twice (native + forced J=J_best) — if their medians diverge by more than `GGML_HIP_TUNE_NOISE_PCT` (default 5%), the run is unreliable. RV21 found the same kernel reading 14% apart at 3 screening samples and 0.6% apart at 15.

2. **Use an idle GPU.** Check `rocm-smi --showuse` first. A `llama-server` on the XTXs will quietly contaminate results.

3. **A contended machine invalidates everything.** RV11, RV15, and RV18 each silently invalidated earlier measurements through different mechanisms: machine contention, wrong RCCL setting, and bench profile producing different shapes than production.

4. **Tuning runs are JSONL at runtime, SQLite offline.** Production never links SQLite (Standards 9.1). The SQLite conversion is always a post-process step.

5. **A cache holds one generation; re-tuning replaces it.** Per-release history and newest-wins are still to design — see HI23.

6. **The dispatch key excludes manifest hash.** A rebuild that changes the catalog doesn't invalidate existing tuning — winners are used with a stale warning. The real guards are per-entry: the loader drops entries naming candidates this binary lacks, and `can_execute` is re-run before launch.

7. **`-DGGML_HIP_RCCL=ON` is required for multi-GPU**. Without it, tensor split falls back to butterfly allreduce and costs 1.5–1.7× end-to-end. The only symptom is one line: `internal AllReduce init failed (n_devices != 2?)`.

8. **Files created on brutus may be invisible from SMB** (`J:` on Windows). Copy them back with `scp` if needed.
