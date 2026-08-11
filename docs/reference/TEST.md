# Test reference

Testing procedures, tuning workflows, dispatch modes, and coverage audits.

See also: [BUILD.md](BUILD.md) — build commands and recipe configuration.

## Offline tests (no GPU)

```bash
cd $BC/tools
python -m unittest discover -s tools/tests      # patcher tests
python3 -m bigcherry audit                      # 32 invariants
python3 -m bigcherry apply --dry-run            # patch placement
python3 -m bigcherry apply                      # idempotent; safe to repeat
python3 -m bigcherry generate --variant-set workload-max \
        --inventory $BC/artifacts/mtp-inventory.json
```

Run all four after touching `src/`, `patches/` or `tools/`.

**If you edit a patch's *text*, `git checkout` its target file first.** The
idempotence guard sees its own output and skips:

```bash
cd $BC/vendor/llama.cpp && git checkout ggml/src/ggml-cuda/mmq.cu
cd $BC/tools && python3 -m bigcherry apply
```

## Correctness

```bash
cd ~/bc-build-multi/bin
HIP_VISIBLE_DEVICES=<n> GGML_HIP_DISPATCH_MODE=native ./test-backend-ops test -o MUL_MAT
HIP_VISIBLE_DEVICES=<n> GGML_HIP_DISPATCH_MODE=tune \
  GGML_HIP_TUNE_SCREEN_SAMPLES=1 GGML_HIP_TUNE_FINAL_SAMPLES=1 \
  ./test-backend-ops test -o MUL_MAT
```

**Always run `native` as well.** It is the baseline that tells you whether a
failure is yours or the hardware's.

A sweep needs a measured **568 MiB** of free VRAM; check before running on a shared card.

## Timing

```bash
GGML_HIP_TUNE_SCREEN_SAMPLES=3 GGML_HIP_TUNE_FINAL_SAMPLES=15   # ~10 min/sweep
```

- **Never draw a performance conclusion from a 1/1 run.** It cannot separate a
  1% difference from noise. Use an idle GPU.
- Other knobs: `GGML_HIP_TUNE_MAX_WORKSPACE`, `GGML_HIP_DISPATCH_DB` (writes
  `<path>.measurements.jsonl` with per-candidate `status`, `median_us`, `nmse` —
  the fastest way to see *why* a candidate was rejected), and
  `GGML_CUDA_DISABLE_GRAPHS=1` (required for complete tuning).

**`GGML_HIP_TUNE_NOISE_PCT` (default 5) — the noise canary.** Native and a forced
MMQ candidate at `J == J_best` are *the same kernel*: the patched switch overwrites
`J_best` with `forced_J` and calls one launcher. Any difference between their
medians is measurement error, and the pair calibrates the harness with no external
reference. Every result records `canary_pct`, `canary_retries` and `canary_pair`
in the measurements JSONL. Check it before believing a narrow margin.

## Coverage

```bash
GGML_HIP_DISPATCH_MODE=replay GGML_HIP_DISPATCH_COVERAGE=cov.json \
llama-bench -m <model> -p 0 -n 64 -r 1 -ngl 99
```

Coverage must read 100% dispatched/executed on token generation.

**`dispatched == executed` does not mean the cache was used.** A miss is still a
dispatch to native, so a fully-covered run can be entirely untuned. Replay builds
now emit provenance:

```json
"total_dispatched": 1188,
"replay": { "entries": 1155, "misses": 1, "stale": true }
```

`misses` needs `GGML_HIP_DISPATCH_MISS=native-record` to be meaningful — without
it the count is always zero. `stale` means winners were measured against a
different candidate set: still valid, possibly no longer best.

## Dispatch modes

Set via `GGML_HIP_DISPATCH_MODE`:

- **`native`** — baseline; no dispatch layer engaged
- **`record`** — emits JSONL to `<path>.measurements.jsonl` via `GGML_HIP_DISPATCH_DB`; build needs `GGML_HIP_AUTOTUNE_RECORD=ON`
- **`tune`** — measures all eligible candidates per signature
- **`replay`** — loads cache from `GGML_HIP_DISPATCH_CACHE`; build needs `GGML_HIP_DISPATCH_REPLAY=ON`

Anything unrecognised warns and falls back to native.

## Getting winners onto the hot path

```bash
# 1. tune (writes <db>.measurements.jsonl)
GGML_HIP_DISPATCH_MODE=tune GGML_HIP_DISPATCH_DB=/tmp/t.jsonl \
  GGML_HIP_TUNE_SCREEN_SAMPLES=3 GGML_HIP_TUNE_FINAL_SAMPLES=15 \
  GGML_CUDA_DISABLE_GRAPHS=1 <llama-server …>

# 2. export
python3 -m bigcherry.replay_cache /tmp/t.jsonl.measurements.jsonl \
  --manifest artifacts/<rev>/hip-autotune-manifest.json --output dispatch.cache

# 3. replay (needs a separate GGML_HIP_DISPATCH_REPLAY=ON build from the SAME manifest)
GGML_HIP_DISPATCH_MODE=replay GGML_HIP_DISPATCH_CACHE=dispatch.cache \
  GGML_HIP_DISPATCH_COVERAGE=cov.json GGML_HIP_DISPATCH_MISS_LOG=miss.jsonl \
  <llama-server …>
```

## Replay slimming (`replay-slim`)

Filters the catalog to the variants a tuning run chose. Order matters — generate
slim *first*, then export the cache against the slim manifest:

```bash
python3 -m bigcherry generate --variant-set replay-slim \
  --inventory artifacts/mtp-inventory.json --winners <db>.measurements.jsonl
python3 -m bigcherry.replay_cache <db>.measurements.jsonl \
  --manifest artifacts/<rev>/hip-autotune-manifest.json --output dispatch-slim.cache
cmake -B build-slim -DGGML_HIP_DISPATCH_REPLAY=ON \
  -DGGML_HIP_AUTOTUNE_VARIANT_SET=replay-slim -DGGML_HIP_AUTOTUNE_SIGNATURE_FILE=…
```

## Replay diagnostics

Optional per-cache-entry replay diagnostics are behind `GGML_HIP_REPLAY_DIAGNOSTICS`.
Production replay builds remain free of hit tracking overhead. Set
`GGML_HIP_DISPATCH_HIT_LOG` in a diagnostic replay build to emit JSONL rows containing
dispatch digest, signature digest, candidate name, and aggregated calls. Shutdown must
use the opt-in `POST /shutdown` endpoint so buffered HIP records are flushed.

## Candidate reference

```bash
python tools/candidate_report.py     # -> docs/reference/CANDIDATES.md
```

Reads the newest manifest plus every log in `artifacts/tuning-logs/`.
