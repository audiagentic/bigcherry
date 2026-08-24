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

## Server benchmark (Brutus bench runner)

End-to-end pp/tg numbers for a **running** llama-server instance come from the
bench harness on Brutus (`ssh brutus` / `10.10.100.10`, key auth, no password).
Harness root: `/mnt/vault/development/llmhosts/llamacpp/bench`. For bigcherry
tests we use **server-bench endpoint mode only** — point it at a server we
started ourselves (tune/replay/native build of our choice); the harness's own
build lanes and spawn mode are not used.

```bash
ssh brutus 'cd /mnt/vault/development/llmhosts/llamacpp && python3 bench/run_bench.py \
  --bench-type server-bench \
  --server-url http://127.0.0.1:42007 \
  --model <label> \
  [--bench-configs default] \
  [--toggles "{\"repetitions\":1}"]'
```

- **`--server-url`** — the running instance. In endpoint mode `--model` is only a
  label for result matching: pass the model's gguf base name to pick up its
  profile/toggles, or any string (e.g. `dummy`) if you want the defaults.
- **`--bench-configs`** — config set from `bench/config/bench-configs.json`
  (`server-bench` section): `default` (pp512 + tg128), `full`,
  `long-prompt-12k`, `long-prompt-16k`, `request-cache`, `mtp-dual`,
  `compression-*`, or comma-separated names (`tg128` for a fast smoke).
- **`--toggles`** — JSON overrides applied last: `repetitions`,
  `prompt_length`, `generation_length`, `ubatch_size`. Use
  `{"repetitions":1}` plus one short config when you only need a liveness check.
- OpenAI-style endpoints: add `--api-type openai` (or use a URL containing
  `/v1`) and `--server-model <name>` for the endpoint's model id.
- Output: per-config `<name>_tps` on stdout; one row appended to
  `bench/results.json` / `results.db`; full log under `bench/raw_logs/` (path
  printed in the output).
- Timing discipline is the same as the offline sweeps: the GPU behind the server
  must be idle, and a short single-repetition run is a liveness check, not a
  performance conclusion.

## Real GPU profiling (rocprofv3)

`tools/bigcherry/rocprof.py` wraps a real server/binary launch under
`rocprofv3 --kernel-trace` and reduces the resulting trace into a
per-kernel-family time breakdown plus real per-GPU busy time (interval
union, not naive sum) — built for HI117's finding that isolated tune-time
candidate measurements do not reliably predict real end-to-end effect;
always verify a kernel-level win against a real profiled trace before
trusting it, not just an isolated benchmark delta.

```python
from bigcherry import rocprof
cmd = rocprof.wrap_command(["./llama-server", "-m", "model.gguf", ...],
                            output_directory=Path("/tmp/profile-out"))
# launch cmd, drive it with real back-to-back requests (no idle gaps —
# the trace span is used as a serving-time proxy), shut it down cleanly
# (rocprofv3 only flushes on normal process exit)
trace = rocprof.find_kernel_trace(Path("/tmp/profile-out"))
families, agents = rocprof.summarize(rocprof.load_kernel_trace(trace))
print(rocprof.format_summary(families, agents))
```

Compare the SAME kernel family (e.g. `mmq`) across two legs built from
identical real driving traffic — that is the only apples-to-apples
comparison; absolute totals depend heavily on how much idle time is in
the trace (utilization % is only meaningful when idle gaps are minimal).

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
