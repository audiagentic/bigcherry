# Build and test reference

Environment setup, build commands, test invocations, and operational procedures.

## Environment — brutus (`10.10.100.10`)

**`~/bigcherry` is the live tree — a real local git clone.** Git operations
(fetch/checkout/status) over the old SMB-mounted path (below) were too slow
and unreliable over the network, so brutus builds from its own local
checkout now, not from the share. This means it can drift from what's
pushed if someone hand-edits there: **treat it as a normal git worktree,
not a build cache** — before building, `git status` must be clean and
`git log -1` must match what you expect (fetch/pull first if not); after
any hands-on edit there, commit and push before moving on, or `git stash`
it. An uncommitted edit left sitting in `~/bigcherry` between sessions is
easy to lose and easy to mistake for already-landed work later (this has
happened at least once: RE27's replay-classification restore sat
uncommitted here for two days after already being re-committed from a
copy pulled in a prior session).

**SMB share — file transfer only, never git/build:**

```text
J:\development\llmhosts\bigcherry  ==  /mnt/vault/development/llmhosts/bigcherry
```

Still useful for copying a build artifact or doc back and forth by hand.
Do not `git` anything against this path, and do not point cmake at it for
a brutus build — use `~/bigcherry` for that.

**Device indices:** 0,1 = gfx1100 (RX 7900 XTX), 2 = gfx1201 (RDNA4), 3 = gfx1030 (RDNA2)

**ROCm at `/opt/rocm`** (check `rocm-smi --showuse` before runs; a running
`llama-server` will contaminate results). cmake 3.28, ninja 1.11, python 3.12.

### SMB traps

- **Dotfiles written on the server are invisible from `J:`** — test with a
  normally-named file, not `.foo`.
- **Files created by a server-side command may be invisible from `J:` entirely.**
  Produce repo files from the Windows side. If a server-side tool generates one,
  copy it back:

  ```bash
  scp 10.10.100.10:/tmp/thing.md docs/reference/THING.md   # run from Windows
  ```

## Standard build cycle

`$BC` = `~/bigcherry` on brutus.

### Linux — all three GPUs

```bash
cd $BC/tools
python3 -m bigcherry audit
python3 -m bigcherry apply
python3 -m bigcherry generate --variant-set workload-max --inventory <inv.json>

cmake -S $BC/vendor/llama.cpp -B ~/bc-build-multi -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGGML_HIP_RCCL=ON \
  -DGGML_HIP_AUTOTUNE=ON -DGGML_HIP_AUTOTUNE_VARIANT_SET=workload-max \
  -DGGML_HIP_AUTOTUNE_SIGNATURE_FILE=$BC/artifacts/mtp-inventory.json \
  -DAMDGPU_TARGETS="gfx1100;gfx1201;gfx1030" -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++

cmake --build ~/bc-build-multi -j
```

**`-DGGML_HIP_RCCL=ON` is not optional for anything multi-GPU.** Without it a
`-sm tensor` run falls back to butterfly allreduce, which sits on the critical
path of every layer and costs **1.5–1.7× end-to-end**. The only symptom is one
line among hundreds:

```text
internal AllReduce init failed (n_devices != 2?); falling back to meta-backend butterfly
```

`workload-max` *requires* `GGML_HIP_AUTOTUNE_SIGNATURE_FILE`; CMake rejects the
combination without it. Useful targets: `ggml-hip` (fastest way to find a compile
error), `test-backend-ops`, `llama-bench`, `llama-server`.

### Windows — workstation's 7900 GRE (gfx1100)

```powershell
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
$env:HIP_PATH = 'C:\Program Files\AMD\ROCm\7.1'
cmake -S 'J:/development/llmhosts/bigcherry/vendor/llama.cpp' -B 'C:/bcw' -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGGML_HIP_AUTOTUNE=ON `
  -DGGML_HIP_AUTOTUNE_VARIANT_SET=workload-max `
  -DGGML_HIP_AUTOTUNE_SIGNATURE_FILE='J:/development/llmhosts/bigcherry/artifacts/mtp-inventory.json' `
  -DAMDGPU_TARGETS=gfx1100 -DLLAMA_BUILD_TESTS=ON `
  -DCMAKE_C_COMPILER="$env:HIP_PATH\bin\clang.exe" `
  -DCMAKE_CXX_COMPILER="$env:HIP_PATH\bin\clang++.exe"
cmake --build C:/bcw --target test-backend-ops -j
```

- **Use a short build directory (`C:\bcw`).** Anything under the scratchpad path
  exceeds the 250-character Windows object-path limit.
- **Put ROCm's `bin` on `PATH` when *running*, not only when building.** Without
  it the exe dies with `0xC0000135` (DLL not found), which looks like a crash.

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
