# Test reference

Testing procedures, tuning workflows, dispatch modes, and coverage audits.

See also: [BUILD.md](../build/BUILD.md) — build commands and recipe configuration.

## Scope and authority

Use this document to choose the right kind of test, not to infer a promotion
verdict from a convenient command. The repository gate and static patch checks
are hardware-free. The correctness, timing, profiling, and RCCL sections are
diagnostics unless they feed the patch's declared Experiment Contract through
the validation/evidence writer. Full contract qualification is opt-in and
hardware-bound; see [PATCH_VALIDATION.md](PATCH_VALIDATION.md).

## Hardware-free repository gate

```bash
cd $BC
PYTHONPATH=tools python -m unittest discover -s tools/tests
PYTHONPATH=tools python -m bigcherry check
PYTHONPATH=tools python -m bigcherry audit
PYTHONPATH=tools python -m bigcherry patch-validate <patch-id>
```

Run this gate after touching `src/`, `patches/`, or `tools/`. It checks the
repository and patch package without requiring a GPU. `patch-validate` checks
the named package's static/evidence state; it does not run hardware
qualification.

For task-specific, hardware-free diagnostics, use an explicit scope:

```bash
cd $BC
PYTHONPATH=tools python -m bigcherry apply --source <source-tree> --dry-run [selection]
PYTHONPATH=tools python -m bigcherry generate --variant-set workload-max \
        --inventory $BC/artifacts/mtp-inventory.json
```

Do not use an unscoped `bigcherry apply` as a generic test. A non-dry apply
mutates a source tree and is an explicitly scoped operation, not an offline
repository gate.

**Never reset or edit the shared vendor tree to retest a patch.** The
idempotence guard correctly skips output it already owns. Inspect existing
validation evidence with:

```bash
cd $BC
PYTHONPATH=tools python -m bigcherry patch-validate <patch-id>
```

To exercise changed patch text, use the explicit
`bigcherry.patch.validation_campaign` workflow, which materializes isolated
content-addressed subject and control trees; it requires a model, HIP toolchain,
manifest, architecture, and dedicated work directory.

For the methodology behind a patch's validation package (README.md +
validation.toml + evidence/validation.json, Experiment Contract binding,
tracked-status semantics), see
[PATCH_VALIDATION.md](PATCH_VALIDATION.md).

### Running a validation campaign on Brutus — working recipe and prerequisites

A real, working invocation (RD73, 2026-09-04). Adapt patch/model/contract
flags; the surrounding setup is what matters:

```bash
cd ~/<isolated-bigcherry-clone>
export PYTHONPATH=tools
export HIP_VISIBLE_DEVICES=0,1        # REQUIRED - see (1)
export ROCR_VISIBLE_DEVICES=0,1
python -m bigcherry.patch.validation_campaign \
  --patch 1233_rd73_stable_graph_cache_key \
  --model /mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf \
  --hip-path /home/audumla/rocm-shim \
  --amdgpu-targets gfx1100 \
  --manifest /home/audumla/bigcherry/artifacts/2578138397d7/hip-autotune-manifest.json \
  --workdir  ~/rd73-contract-run \
  --build-root ~/rd73-contract-builds \
  --worktree-root ~/rd73-contract-worktrees \
  --run-rd73-contract \
  --rd73-corpus tools/bigcherry/bench/corpora/mtp-27b-v1.jsonl
```

The command above is Brutus-specific qualification infrastructure, not a
portable repository interface. `/home/audumla`, `/mnt/vault`, the model path,
and the work directories must be replaced on another host. A run contributes
qualification evidence only when the campaign persists the required contract
identity, measurements, provenance, and verdict through the canonical evidence
path.

Three prerequisites that are easy to miss and each cost a failed run:

1. **`HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` must both be exported.**
   The campaign deliberately inherits ambient GPU visibility rather than
   restricting it (so `-sm tensor` topology is preserved) and fails closed if
   they are unset. On this box, running without them exposes all four
   heterogeneous GPUs; the server then logs
   `internal AllReduce init failed (n_devices != 2?)` and **segfaults**
   (`server process exited (code -11) before becoming healthy`).

2. **`--hip-path` must point at a tree whose `bin/` contains `clang`/`clang++`.**
   The campaign builds its compiler paths as `<hip-path>/bin/clang`. This
   ROCm install exposes `amdclang`/`amdclang++` in `/opt/rocm/bin` and puts
   real `clang` under `/opt/rocm/llvm/bin`, so `--hip-path /opt/rocm` fails
   configure with *"is not a full path to an existing compiler tool"*. Build a
   non-invasive shim rather than modifying `/opt/rocm`:

   ```bash
   R=~/rocm-shim; rm -rf $R; mkdir -p $R/bin
   for e in /opt/rocm/*;     do b=$(basename $e); [ "$b" = bin ] && continue; ln -s $e $R/$b; done
   for e in /opt/rocm/bin/*; do ln -s $e $R/bin/$(basename $e); done
   ln -sf /opt/rocm/llvm/bin/clang   $R/bin/clang
   ln -sf /opt/rocm/llvm/bin/clang++ $R/bin/clang++
   ```

   `ROCM_PATH`/`HIP_PATH`/`CMAKE_PREFIX_PATH` all resolve correctly through the
   shim because every other top-level entry is symlinked straight to
   `/opt/rocm`.

3. **Never run `cmake --build` by hand inside `~/.cache/bigcherry/builds/...`.**
   Those directories are identity-bound. Adding a target manually (e.g.
   building `llama-server` yourself) makes the next campaign fail with
   *"has an existing bigcherry-build-metadata-llama-bench.json but it failed
   reuse validation -- refusing to silently rebuild over it: runtime bundle
   hash does not match recorded identity"*. The campaign builds `llama-server`
   and `llama-bench` for both control and validation-subject lanes itself, so
   this is never necessary. If a directory has already been contaminated,
   delete that build directory and let the campaign rebuild it.

**Do not substitute an ad-hoc A/B for the contract.** A hand-run
control-vs-subject comparison with one completion per arm cannot resolve a
low-single-digit effect against this project's measured 0.5-0.9% repetition
noise floor. RD73 is the worked example: an ad-hoc single-sample run was read
as "null, mechanism inert", while the contract's 10 paired rounds with 10,000
bootstrap resamples measured **+1.855%, 95% CI [1.482%, 2.169%]**. That is a
historical measurement, not the current acceptance decision: the live RD73
contract requires the 95% CI lower bound for gain to be at least 1.0%, the
95% CI upper bound for control regression to be at most 1.0%, and at least 10
paired rounds under `ci95_threshold_bound_v1`. Resolve thresholds from
`config/experiment-contracts.toml`; do not copy them into a patch README.
Use the contract path for any claim, positive or negative.

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

## Heterogeneous RCCL source qualification

For heterogeneous-architecture RCCL source diagnosis, repair qualification, crash-isolated candidate testing, topology identity, and eventual collective replay/tuning, use [RCCL_HETEROGENEOUS_RUNBOOK.md](RCCL_HETEROGENEOUS_RUNBOOK.md).

Do not use ordinary tuning sweeps to rediscover HI85's established heterogeneous RCCL crash behavior. Historical failures are immutable evidence for their exact tested topology, device set, build, and runtime. They do not establish a universal heterogeneous-architecture prohibition, and patch 1225 is an earlier, scoped guard design—not proof of complete or current protection. Verify the actual patch composition and shared-admission implementation before treating source-level viability as established; only then is heterogeneous RCCL performance tuning or replay eligible.

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
PYTHONPATH=tools python -m bigcherry.analysis.candidate_report  # -> docs/reference/CANDIDATES.md
```

Reads the newest manifest plus every log in `artifacts/tuning-logs/`.
