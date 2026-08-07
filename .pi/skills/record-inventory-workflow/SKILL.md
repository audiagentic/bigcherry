---
name: record-inventory-workflow
description: Record HIP dispatch observations from a running server on brutus, build an inventory JSON and SQLite database, and use them to generate a workload-tuned candidate catalog. Use when the user wants to capture which kernels are actually called by a real workload.
---

# Record-mode observation workflow

This skill walks through capturing kernel dispatch observations from a running llama.cpp server and converting them into an inventory that drives the autotune candidate catalog.

## Prerequisites

- SSH access to brutus (`audumla@10.10.100.10`)
- Pre-built multi-arch binary at `~/bc-build-multi/bin/llama-server` (already configured for gfx1100, gfx1201, gfx1030)
- A model to run — see "Model paths" below

## Model paths on brutus

| Workload | Path |
| --- | --- |
| Qwopus 27B MTP (target config) | `/mnt/vault/llm-models/qwen3.6-27b/gguf/mtp/Qwopus3.6-27B-v2-MTP-Q8_0.gguf` |
| Qwen 2B IQ4_XS (quick test, single GPU) | `/mnt/vault/llm-models/qwen3.5-2B/gguf/Qwen_Qwen3.5-2B-IQ4_XS.gguf` |

## Step 0 — Find a free GPU on brutus

```bash
rocm-smi --showuse
```

Device indices: **0,1 = gfx1100 XTX (tensor-split pair), 2 = gfx1201, 3 = gfx1030 (RDNA2)**. Pick one with 0% VRAM usage. For the 27B model, you need both XTXs (devices 0+1). For quick single-GPU tests, use device 3 (gfx1030) — it's usually free.

## Step 1 — Record dispatches

### Target config (2× XTX tensor split with MTP)

```bash
GGML_HIP_DISPATCH_MODE=record \
GGML_HIP_DISPATCH_DB=/tmp/rec.jsonl \
~/bc-build-multi/bin/llama-server -m /mnt/vault/llm-models/qwen3.6-27b/gguf/mtp/Qwopus3.6-27B-v2-MTP-Q8_0.gguf \
  --host 127.0.0.1 --port 42099 \
  -dev ROCm0,ROCm1 -sm tensor -ts 1,1 -ngl 99 \
  -c 16384 -ctk f16 -ctv f16 -fa auto -b 2048 -ub 512 -t 8 \
  --spec-type draft-mtp --spec-draft-n-max 5 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 &
```

### Quick test (single GPU, gfx1030)

```bash
HIP_VISIBLE_DEVICES=3 \
GGML_HIP_DISPATCH_MODE=record \
GGML_HIP_DISPATCH_DB=/tmp/rec.jsonl \
~/bc-build-multi/bin/llama-server -m /mnt/vault/llm-models/qwen3.5-2B/gguf/Qwen_Qwen3.5-2B-IQ4_XS.gguf \
  --host 127.0.0.1 --port 42098 \
  -dev ROCm0 -ngl 99 \
  -c 2048 -ctk f16 -ctv f16 -fa auto -b 512 -ub 256 -t 4 &
```

The server writes `/tmp/rec.jsonl.measurements.jsonl` with one JSON line per dispatch.

**Use `llama-server`, not `llama-bench`.** A server with MTP enabled produces draft widths `[1,2,4,5,6,8,...]`; `llama-bench` produces `widths: [1]` only (17 vs 80+ signatures).

## Step 2 — Drive the server with a benchmark

```bash
python3 /mnt/vault/development/llmhosts/llamacpp/bench/run_bench.py \
  --bench-type server-bench --server-url http://127.0.0.1:42099 \
  --model /mnt/vault/llm-models/qwen3.6-27b/gguf/mtp/Qwopus3.6-27B-v2-MTP-Q8_0.gguf \
  --timeout 300 --upload-dry-run
```

This sends enough requests to exercise the server and populate the record file.

## Step 3 — Stop the server

```bash
kill %1  # or find and kill by port: fuser -k 42099/tcp
```

## Step 4 — Build inventory and SQLite

```bash
cd /mnt/vault/development/llmhosts/bigcherry
python3 -m bigcherry inventory record /tmp/rec.jsonl \
    --inventory artifacts/my-inventory.json \
    --database artifacts/my-inventory.sqlite
```

This produces:

- **Inventory JSON** — summary of observed types, widths, and BLAS usage (consumed by `generate --variant-set workload-max`)
- **SQLite database** — structured observation data with call counts per signature

## Step 5 — Generate the catalog

```bash
cd /mnt/vault/development/llmhosts/bigcherry
python3 -m bigcherry generate --variant-set workload-max \
    --inventory artifacts/my-inventory.json \
    --arch gfx1100,gfx1201
```

This reads the inventory and generates a candidate catalog tailored to the observed workload. The `workload-max` variant set expands the candidate space based on observed widths and types — it requires an inventory file.

If you have tuning winners from a previous run, you can also generate a `replay-slim` set:

```bash
python3 -m bigcherry generate --variant-set replay-slim \
    --inventory artifacts/my-inventory.json \
    --winners /tmp/tune.measurements.jsonl \
    --arch gfx1100
```

Order matters: generate slim first, then export the cache.

## Gotchas

1. **Device lists differ between tools**: `llama-bench` uses `/` (e.g., `-dev ROCm0/ROCm1`), `llama-server` uses `,` (e.g., `-dev ROCm0,ROCm1`). Wrong separator means two single-GPU runs instead of one tensor-split run.

2. **`-DGGML_HIP_RCCL=ON` is required for multi-GPU**. Without it, tensor split falls back to butterfly allreduce and costs 1.5–1.7× end-to-end. The only symptom is one line: `internal AllReduce init failed (n_devices != 2?)`.

3. **Files created by server-side commands may be invisible from SMB**. If a Python script writes a file on brutus, copy it back with `scp` rather than relying on SMB visibility.

4. **The observation database is separate from tuning**. Record mode produces an observation DB; tuning produces a measurement DB. They have different schemas but can be merged into one SQLite file for the `hot` report. Load observations first (`inventory record`), then overlay tuning (`inventory tuning`).

5. **Editing a patch's text on an already-patched tree is a no-op**. The idempotence guard skips it. `git checkout` the target file first.

6. **Use a short build directory on Windows** (e.g., `C:\bcw`). Anything under the long scratchpad path exceeds the 250-character Windows object-path limit.
