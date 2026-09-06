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

### Running a validation campaign on the build server — working recipe and prerequisites

A real, working invocation (RD73, 2026-09-04). Adapt patch/model/contract
flags; the surrounding setup is what matters:

```bash
cd ~/<isolated-bigcherry-clone>
export PYTHONPATH=tools
export HIP_VISIBLE_DEVICES=0,1        # REQUIRED - see (1)
export ROCR_VISIBLE_DEVICES=0,1
python -m bigcherry.patch.validation_campaign \
  --patch 1233_rd73_stable_graph_cache_key \
  --model $BC_MODEL_ROOT/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf \
  --hip-path $BC_ROCM_SHIM \
  --amdgpu-targets gfx1100 \
  --manifest $BC_REPO/artifacts/2578138397d7/hip-autotune-manifest.json \
  --workdir  ~/rd73-contract-run \
  --build-root ~/rd73-contract-builds \
  --worktree-root ~/rd73-contract-worktrees \
  --run-rd73-contract \
  --rd73-corpus tools/bigcherry/bench/corpora/mtp-27b-v1.jsonl
```

The command above is build-server qualification infrastructure, not a
portable repository interface. Resolve the paths with
`source tools/env/bigcherry-env.sh` (see [ENVIRONMENT.md](../ENVIRONMENT.md));
on another host, change `config/environment.toml` rather than this command. A run contributes
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
low-single-digit effect against this project's measured repetition noise.
RD73 is the worked example: an ad-hoc single-sample run was read as "null,
mechanism inert", and that conclusion was wrong -- the contract path later
measured a real, repeatable gain. Use the contract path for any claim,
positive or negative.

**One run's confidence interval is not the uncertainty of the result.**
RD73 again, on real hardware: six sessions of the SAME unchanged build, same
corpus, same host, measured

    1.326  2.356  1.373  1.922  2.626  1.730   (% end-to-end)

a between-session sd of ~0.52 -- LARGER than the standard error any single
run reported -- with one session's point estimate falling below another's
`ci95_low`. Every one of those runs was honest. `block_bootstrap_effect()`
resamples pairs WITHIN a run, so its interval covers only within-session
variation and is blind to drift between occasions. Quoting one run's interval
therefore overstates precision.

Where a claim has to hold across occasions, measure across sessions:
`bootstrap_session_effect()` (a two-level bootstrap that resamples whole
sessions, minimum four) and `aggregate_session_effects()`, which rebuilds the
interval from the `lane_effects` persisted in each record. RD73's promotion
used exactly this: 6 sessions, **+1.889%, 95% CI [1.475, 2.352]**, 0.0%
control regression.

Those numbers are a historical measurement, not a current acceptance
decision. Resolve live thresholds and evidence policy from
[`config/experiment-contracts.toml`](../../../config/experiment-contracts.toml);
do not copy scientific thresholds into this guide or a patch README.

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

## Server benchmark (build-server bench runner)

End-to-end pp/tg numbers for a **running** llama-server instance come from the
bench harness on the build server (`ssh $BC_HOST`, key auth, no password).
Harness root: `$BC_BENCH_HARNESS`. For bigcherry
tests we use **server-bench endpoint mode only** — point it at a server we
started ourselves (tune/replay/native build of our choice); the harness's own
build lanes and spawn mode are not used.

```bash
source tools/env/bigcherry-env.sh
ssh "$BC_HOST" "cd $BC_BENCH_HARNESS/.. && python3 bench/run_bench.py \
  --bench-type server-bench \
  --server-url http://127.0.0.1:$BC_BENCH_PORT \
  --model <label> \
  [--bench-configs default] \
  [--toggles '{\"repetitions\":1}']"
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

## Comparative A/B benchmarking — mandatory procedure

Everything above tells you how to get a number. This section is about
comparing two numbers, which is where every real mistake has happened. Each
rule below exists because it was broken, and each time the result was reported
before the defect was found. Treat them as required, not advisory.

**Use `bigcherry ab-benchmark`.** It is paired and interleaved, takes
`--pairs`, `--schedule-seed`, `--settle-seconds`, `--practical-threshold-pct`
and `--decision-grade`, and checks common CMake settings through
`--stock-cmake-cache`/`--patched-cmake-cache`. This is not complete build or
measurement admission: the caller must still prove source/binary identity,
activation, work equivalence, and clean teardown. Its command-mode CLI does
not yet orchestrate a server per arm.

Do not write a new harness. `tools/lab/gp11-replay-bench/ab-balanced.sh` was
written from scratch in ignorance of `ab-benchmark`, reimplemented order
balancing badly, and produced a result that had to be retracted for a confound
`ab-benchmark` already handles. It survives only as a record of that.

For the questions a throughput A/B cannot answer:

| question | command |
|---|---|
| what fraction of time does each kernel take | `bigcherry kernel-fraction` over rocprofv3 kernel-trace CSVs |
| what saving should tuning predict | `bigcherry impact --observations <record jsonl> --measurements <promoted.jsonl>` |
| repeatable deep profiling | `bigcherry profile-campaign` (docs/reference/tooling/PROFILING.md) |
| which candidate served which dispatch | `[build.replay-diagnostic]` + `GGML_HIP_DISPATCH_HIT_LOG` |

### Reusable campaign build matrix (HI168)

Build via the existing campaign engine, selecting the model/topology's own
inventory and promoted winners. The profile builds servers, not llama-bench:

```bash
PYTHONPATH=tools python3 -m bigcherry build --profile e2e-build-matrix \
  --arch gfx1100 --inventory <campaign>/inventory.json \
  --winners <campaign>/promoted.jsonl
```

`--arch` must match the topology; it is not a device-visibility selector.
Do not build alongside measurement. Existing compatible content-addressed
builds may be reused; never alter those build trees by hand.

| Build | Runtime role | Diagnostic content | Interpretation |
| --- | --- | --- | --- |
| stock, llama-native source | native | no BC dispatch | genuine upstream baseline |
| native, bigcherry-native source | native | tuner/dispatch diagnostics OFF | framework-only production-shaped baseline |
| control | native | AUTOTUNE implies diagnostics | instrumented control, not production native |
| record | record | tuner and recording/diagnostics | signature observation, not production timing |
| tune | tune | tuner, diagnostics, workspace metrics | tuning observations, not production timing |
| replay | native and replay | diagnostics OFF | same-binary winner-effect pair |
| replay-diagnostic | native and replay | dispatch counters and hit-log attribution | diagnostic companion, not production timing |
| audit (opt-in, outside profile) | compile audit | all candidates, tuner/diagnostics | unnecessary for ordinary workload-specific E2E |

The new `native` build has an inventory-only native catalog. Do not substitute
it for the native arm of a same-binary replay comparison. Source composition
(`bigcherry-native` framework vs `bigcherry` release) remains separate from
build type and runtime mode. Differences against stock describe the whole
composition; they do not isolate one patch.

Inspect every candidate build without starting hardware:

```bash
PYTHONPATH=tools python3 -m bigcherry ab-benchmark --inspect-build <build-directory>
```

This reports declared CMake flags alongside actual compiler definition counts
and coverage-TU presence. A cache value of diagnostics OFF is insufficient:
AUTOTUNE enables that compiler definition implicitly. The command returns 1
for detected coverage/diagnostic disagreement and 2 for unreadable inputs.
An exit 0 establishes only this limited compiler-configuration observation,
not binary integrity, full production eligibility, or runtime activation.
Use campaign completed-build identities and inspect the actual library too.

For every future model/topology run retain its model identity, effective
runtime profile, both device selectors, source/build/runtime-bundle identities,
bench-config content, cache digest, per-cell order and metrics, work-equivalence
evidence, and shutdown result. Production timing and diagnostic observations
must be stored with explicit roles; never transfer a companion's throughput
to a production claim or treat companion activation as same-cell proof.
The server-per-arm integration and production activation admission remain
HI168 work; this profile alone is not an executable benchmark campaign.

### 1. Never run arms in a fixed order

Running arm A then B then C, one sample each, confounds arm with position:
thermal drift over a ~10-minute run produces a clean-looking monotone result
that has nothing to do with the arms. Measured position spread on the build server is
0.19–0.35%, which is the same order as the effects being chased.

Rotate the arm order every round, use a round count that is a multiple of the
arm count, and **record the position on every result row** so drift can be
tested rather than assumed away.

### 2. Verify what is actually IN each binary before interpreting anything

Never trust a build digest you noted earlier, or a label like "control". Read
the composition out of the build itself:

```bash
# which source tree did this build compile?
python3 -c "
import json; cc=json.load(open('<build>/compile_commands.json'))
print([e['file'] for e in cc if 'ggml-cuda' in e['file']][0].split('/ggml/')[0])"

# what is in that tree, and what flags were set?
grep -rlq '<marker for the patch>' <src>/ggml/src/
grep -a 'GGML_HIP_DISPATCH_REPLAY:BOOL=' <build>/CMakeCache.txt
```

Two arms intended to differ by one build flag must have **byte-identical
source trees**; hash them and check:

```bash
find <src>/ggml/src -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' \
  | sort | xargs cat | sha256sum
```

A comparison whose arms differ in more than one way cannot attribute its own
result, no matter how clean the numbers are.

### 3. Shut the server down gracefully — never `kill -9`

`kill -9` skips backend teardown. That discards buffered HIP autotune
measurements, **and** it destroys the replay hit/miss report and the coverage
report, which are emitted at shutdown.

The `/shutdown` route comes from patch `0800_server_shutdown_endpoint`, and it
is **registered only when `LLAMA_SERVER_ENABLE_SHUTDOWN` is set in the
server's environment**. Without that variable the POST 404s and you are back
to `kill -9` without noticing. Set it at launch, and treat a failed POST as a
loud error, not a fallback.

Prefer `tools/bigcherry/tuning/server_runner.py` (`ServerRunner`) over hand-
rolled bash: it already handles launch, health-check, env hygiene
(`env_unset`), free-port selection, and graceful teardown. Every divergence
from it in an ad-hoc harness has so far reintroduced a bug it had already
fixed.

`ServerRunner.shutdown()` returns a `ShutdownResult` (also retained as
`last_shutdown`): method, request success, forced-kill flag, exit code and any
request error. Require `clean` before admitting a cell; existing callers that
ignore this result do not gain that gate automatically. For genuine unpatched
stock on POSIX use `shutdown_method="sigint"`, which invokes upstream's normal
signal handler. This mode rejects process wrappers and Windows rather than
pretending a signal reached the intended server. BigCherry defaults to HTTP.

### 4. Prove the mechanism under test was actually active

A cache arm is not a cache arm because you set the variable. Capture both:

- startup: `bigcherry: replay cache '<path>' loaded, N winner(s)`
- shutdown: `bigcherry:   replay v2 N winner(s); exact=.. miss=..`

Without these, "the winners made no difference" and "the cache never resolved
a lookup" are the same measurement. Note also that coverage
`dispatched == executed` does **not** prove tuning was applied — a miss is
still a dispatch, to native, so a fully-covered run can be entirely untuned.

These startup/shutdown requirements apply to an instrumented diagnostic
cell. Production replay intentionally compiles out this telemetry, as
specified in [BUILD.md — diagnostics split](../build/BUILD.md#which-build-to-measure-on-the-diagnostics-split).
Do not enable counters in the production timing build to satisfy this check.

Keep two explicit evidence roles: production cells supply throughput;
diagnostic cells supply activation observations. A diagnostic companion is
not proof that a particular production cell executed the winners. Linking
them requires verified source/generated-input/cache/workload parity and an
explicitly bounded inference. Until that admission path is implemented and
verified, captured production numbers alone do not establish a tuning
benefit. For instrumented causal patch tests, activation still belongs to
the cell being qualified.

### 5. Confirm both arms did the same work

For MTP speculative decode, record **draft acceptance** per cell. If
acceptance differs between arms they generated different amounts of work and
a throughput comparison between them is meaningless regardless of how tight
the intervals look. Identical acceptance is what makes the comparison legal.

### 6. Statistics

- Complete separation (every sample of one arm beyond every sample of the
  other) at n=6 per arm is Mann-Whitney p ≈ 0.0022. Prefer it to a t-test:
  samples are small and between-session drift is not obviously normal.
- **Do not treat "all N metrics moved the same way" as significance.** The
  metrics are strongly correlated; 6/6 is not p = 1/64.
- Between-session drift on this host is sd 0.5–0.6%, larger than any single
  run's standard error. Comparisons across runs need that margin; comparisons
  within one balanced run do not.
- `pp256` is startup-sensitive and routinely throws outliers ~10% low. Exclude
  it from diagnosis; `tg512`/`tg2048` are the stable signals.

### 7. Use the documented harness

`bench/run_bench.py --bench-type server-bench` as described above. Not
`llama-bench` — it cannot see MTP speculative decode at all, so it silently
measures a different thing than production runs.

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

### The log channel does not work under llama-server

**llama-server installs a log callback that swallows the library's
`GGML_LOG_INFO` lines entirely.** The startup `replay cache '<path>' loaded, N
winner(s)` line, the shutdown counter report and the coverage summary never
reach stdout or stderr. A run that looks completely silent may be working
perfectly — this cost hours once, and the silence was misread as "the dispatch
layer never ran".

`GGML_HIP_DISPATCH_COVERAGE=<path>` is the only reliable channel. It is written
from the same flush hook (anchored at `ggml_backend_cuda_free`, so the server
must shut down gracefully) and carries the replay provenance plus, in a
diagnostics build, a `dispatch` object with the hot-path counters.

### `exact > 0` is not proof a tuned kernel ran

After an exact cache hit the resolver still revalidates the candidate —
`can_execute`, architecture support, blacklist, transform applicability — and
can substitute native. A run can therefore report exact hits and launch native
for every one of them.

The only sufficient evidence is `final_tuned_launches > 0`, counted at the
executor after every validation:

```json
"dispatch": { "final_tuned_launches": 0, "final_native_launches": 190950 }
```

That example is real: a valid, non-stale cache with 18 promoted winners, and
**not one tuned kernel executed**. Reading `exact` alone would have called it a
working replay arm.

Requires a `GGML_HIP_DISPATCH_DIAGNOSTICS=ON` build — see
[BUILD.md](../build/BUILD.md), "Which build to measure on". Take the timings
from the production build and this evidence from a diagnostics build of the
same revision; never mix.

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
