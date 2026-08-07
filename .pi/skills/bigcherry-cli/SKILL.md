---
name: bigcherry-cli
description: Reference for the bigcherry command-line tool — pull, audit, apply, generate, status, inventory, report. Use when the user needs to know what commands are available or their arguments.
---

# Bigcherry CLI reference

The `bigcherry` tool manages the full workflow of integrating a new llama.cpp release: pulling upstream, auditing it, applying patches, generating the autotune catalog, and analyzing tuning results.

On brutus, commands run as `python3 -m bigcherry <command>` from `/mnt/vault/development/llmhosts/bigcherry`. On this Windows checkout, run from `J:\development\llmhosts\bigcherry` with `PYTHONPATH=tools` (PowerShell: `$env:PYTHONPATH='tools'`) and use `python -m bigcherry <command>`.

## Core workflow

### pull — clone or update the llama.cpp checkout

```bash
python3 -m bigcherry pull [--ref <tag|sha>] [--full]
```

- `--ref` — tag (e.g., `b1234`), branch, or SHA to check out
- `--full` — full clone instead of depth-1 (needed for arbitrary older revisions)

Clones or fetches into `vendor/llama.cpp`. Records the revision in the release state machine.

### audit — verify upstream invariants

```bash
python3 -m bigcherry audit [--no-strict] [-v]
```

Checks 32 source-level invariants (patchable code, no generated files, etc.). `--strict` (default) treats warnings as failures; `--no-strict` allows warnings. Writes `source-audit.json`.

### apply — apply patches and overlay files

```bash
python3 -m bigcherry apply [--dry-run] [--force]
```

Applies the 16 file patches from `patches/` and copies `src/` overlay files onto the checkout. Refuses to run on an unpatched tree without `--force`.

**If you edit a patch's text, `git checkout` its target file first** — the idempotence guard skips unchanged files.

### generate — generate the autotune candidate catalog

```bash
python3 -m bigcherry generate --variant-set <set> [--arch <archs>] \
    [--inventory <inv.json>] [--winners <winners.jsonl>] [--dry-run] [--force]
```

- `--variant-set` — one of: `inventory`, `workload-max`, `replay-slim`
  - `inventory` — baseline catalog from config tables only
  - `workload-max` — expanded catalog based on observed workload (requires `--inventory`)
  - `replay-slim` — minimal catalog of winners only (requires `--winners`)
- `--arch` — comma-separated architectures or group names (default: `all`)
- `--inventory` — inventory JSON from record mode (required for `workload-max`)
- `--winners` — measurements JSONL from tuning run (required for `replay-slim`)

### status — show checkout and release state

```bash
python3 -m bigcherry status
```

Shows current revision, release stage (pulled → audited → patched → generated → built), and audit/manifest hashes.

## Inventory commands

### inventory record — convert record-mode JSONL to SQLite + inventory JSON

```bash
python3 -m bigcherry inventory record <record.jsonl> \
    [--inventory <path>] [--database <path>]
```

Reads the JSONL from a `GGML_HIP_DISPATCH_MODE=record` run and produces:

- Inventory JSON (summary of observed types, widths, BLAS usage)
- SQLite database (structured observation data with call counts)

Output paths default to alongside the input file (`.inventory.json`, `.sqlite` extensions).

### inventory tuning — load tuning measurements into SQLite

```bash
python3 -m bigcherry inventory tuning <measurements.jsonl> \
    [--database <path>] [--manifest <manifest.json>]
```

Reads the JSONL from a `GGML_HIP_DISPATCH_MODE=tune` run and writes three tables: `winner`, `measurement`, `candidate`. The `--manifest` flag populates full candidate data; without it, you get minimal metadata.

## Report commands

### report summary — aggregate statistics

```bash
python3 -m bigcherry report summary [--measurements <jsonl>] [--database <sqlite>] [--json]
```

Shows: total signatures, improvement distribution (>10%, >5%, >1%), family winner breakdown, rejection counts.

### report signatures — per-signature detail tables

```bash
python3 -m bigcherry report signatures \
    [--measurements <jsonl>] [--database <sqlite>] \
    [--dispatch <hex>] [--limit <N>] [--json]
```

Detailed candidate table per dispatch with median_us, MAD, P95, NMSE, workspace. `--dispatch` filters to one digest; `--limit` caps output.

### report families — cross-family comparison

```bash
python3 -m bigcherry report families --dispatch <hex> \
    [--measurements <jsonl>] [--database <sqlite>] [--json]
```

Groups candidates by family for a specific dispatch, showing which family won and by how much.

### report hot — top-N signatures by call count

```bash
python3 -m bigcherry report hot --database <sqlite> \
    [--limit <N>] [--json]
```

Requires the observation table (from `inventory record`). Ranks signatures by call count, showing which wins matter most for real throughput. Requires a database that has both observation and winner tables.

## Replay cache

```bash
python3 -m bigcherry.replay_cache <measurements.jsonl> \
    --manifest <manifest.json> --output dispatch.cache
```

Builds a binary replay cache from tuning measurements. This is what you load with `GGML_HIP_DISPATCH_CACHE=dispatch.cache` in production.

## Environment variables for llama-server

| Variable | Purpose |
| --- | --- |
| `GGML_HIP_DISPATCH_MODE=native\|record\|tune\|replay` | Dispatch layer mode |
| `GGML_HIP_DISPATCH_DB=/tmp/rec.jsonl` | Record/tune output path (writes `.jsonl`) |
| `GGML_HIP_TUNE_SCREEN_SAMPLES=3` | Screening samples (default 3) |
| `GGML_HIP_TUNE_FINAL_SAMPLES=15` | Final samples for winners (default 15) |
| `GGML_CUDA_DISABLE_GRAPHS=1` | Required for tuning; graphs skip tune |
| `GGML_HIP_DISPATCH_CACHE=dispatch.cache` | Path to replay cache |
| `GGML_HIP_DISPATCH_COVERAGE=cov.json` | Coverage output file |
| `GGML_HIP_DISPATCH_MISS_LOG=miss.jsonl` | Miss log path |
| `GGML_HIP_DISPATCH_MISS=native-record` | Record misses to JSONL |
| `HIP_VISIBLE_DEVICES=3` | Restrict to specific GPU(s) |

## Build prerequisites

- Pre-built multi-arch build: `~/bc-build-multi/` (gfx1100, gfx1201, gfx1030, RCCL enabled)
- Windows local build: `C:\bcw\bin`, ROCm 7.1 at `C:\Program Files\AMD\ROCm\7.1`; build missing benchmark binary with `cmake --build C:\bcw --target llama-bench -j 8`
- ROCm at `/opt/rocm` (7.2 on brutus), cmake 3.28, ninja 1.11, python 3.12
- Bigcherry repo: `/mnt/vault/development/llmhosts/bigcherry` or `J:\development\llmhosts\bigcherry`

## Build cycle

```bash
cd /mnt/vault/development/llmhosts/bigcherry/tools
python3 -m bigcherry audit
python3 -m bigcherry apply
python3 -m bigcherry generate --variant-set workload-max \
    --inventory artifacts/mtp-inventory.json
cmake --build ~/bc-build-multi --target llama-server -j
```
