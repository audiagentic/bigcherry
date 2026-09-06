# Build reference

Environment setup and build commands for bigcherry recipes and manual configuration.

See also: [TEST.md](../testing/TEST.md) — testing, tuning workflows, and coverage.

## Start here: production end-to-end comparisons

Use the campaign engine and server test bench, not the manual CMake or
`llama-bench` examples below. The maintained procedure and build-role matrix
are in [TEST.md — reusable campaign build matrix](../testing/TEST.md#reusable-campaign-build-matrix-hi168).

1. Verify the intended committed revision and a clean local build-server checkout
   (an explicitly isolated worktree is also supported; never disturb another
   session's live tree).
2. Build `e2e-build-matrix` with the model/topology's inventory, promoted
   winners, and architecture. A failed admission or lane is a failed build;
   a successful stock lane does not establish BC build success.
3. Verify completed-build identities and actual compiled diagnostic flags.
   Do not add targets or edit generated inputs inside cached build trees.
4. Once compilation has stopped and the selected GPUs are idle, drive each
   managed server with `bench/run_bench.py --bench-type server-bench` in
   endpoint mode. Follow TEST.md for balancing, provenance and teardown.

Framework configuration qualification is a prerequisite build/evidence
operation, not a substitute for this production performance comparison.

## Environment — the build server

Host facts (address, paths, toolchain, ports, device inventory) live in
[`config/environment.toml`](../../../config/environment.toml); `source
tools/bigcherry-env.sh` exports them as `$BC_*`. See
[ENVIRONMENT.md](../ENVIRONMENT.md).

**`~/bigcherry` is the live tree — a real local git clone.** Git operations
(fetch/checkout/status) over the old SMB-mounted path (below) were too slow
and unreliable over the network, so the build server builds from its own local
checkout now, not from the share. Treat it as a normal git worktree, not a
build cache: before building, `git status` must be clean and `git log -1`
must match what you expect (fetch/pull first if not); after any hands-on
edit there, commit and push before moving on.

**SMB share — file transfer only, never git/build:**

```text
J:\development\llmhosts\bigcherry  ==  $BC_SHARE/bigcherry
```

Still useful for copying a build artifact or doc back and forth by hand.
Do not `git` anything against this path, and do not point cmake at it for
a build-server build — use `$BC_REPO` for that. (This section previously said
the reverse — that `$BC_REPO` was the stale copy and the share was
live. That was wrong and out of date; see BUILD_AND_TEST.md, which already
had the corrected version. If you're reading this after finding it
contradicted `~/bigcherry`'s actual state again, trust the live checkout,
not either doc, and fix whichever doc is behind.)

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
  scp "$BC_HOST:/tmp/thing.md" docs/reference/THING.md   # run from Windows
  ```

## Sources — the normal way to build

`$BC` = `$BC_REPO` on the build server.

A **source** (`[source.<name>]` in `recipes.toml`) names one complete patch composition: an upstream ref, an overlay flag, and the exact `patch-set`(s) it applies. Builds and platforms are a separate, orthogonal axis, composed per-lane.

### Axis table

| Axis | Meaning | Scope | Examples |
|------|---------|-------|----------|
| **Source** | One exact, curated patch composition | Global; names a row in `[source.*]` | `llama-native`, `bigcherry-native`, `bigcherry` |
| **Build** | A cmake variant set | Named independently, composed per-lane | `record` (measures signatures), `tune` (tunes candidates), `replay` (applies winners) |
| **Platform** | GPU target(s) and compile flags | Named independently, composed per-lane | `linux-multi` (3 GPUs on the build server), `windows-gfx1100` (workstation) |
| **Patch state** | Patch acceptance status | Per-patch metadata, informational only under v2 | `validated`, `untested`, `rejected` |

A source's `patch-sets` list is exact and curated -- there is no groups/states predicate filtering axis to override; a source either names a patch-set or it doesn't.

### Repinning

```bash
cd $BC/tools && python3 -m bigcherry repin [--ref <ref>]
```

Rewrites `recipes.toml`'s top-level `pinned = "..."` line in place, leaving comments intact (omit `--ref` to query for the newest upstream release). Sources with `ref = "pinned"` (the default) now build from the new ref; sources naming an explicit ref remain frozen and do not move. This lets you keep a historical source around for comparison without the file bloating.

**Note:** `--source` (below) selects *which patches get applied to the shared checkout* — it's the real, current flag on `audit`/`apply`/`patches`. `build` itself doesn't take it -- it moved to the campaign engine's isolated, content-addressed per-lane sources, where each lane names its exact patch set directly rather than mutating one shared checkout.

### Patch state semantics

- **`validated`** — Measured, proven correct against the baseline (native).
- **`rejected`** — Measured or reviewed, rejected (too risky, no benefit, etc.).
- **`untested`** — New or awaiting measurement.

Under v2, patch state is informational metadata on the patch itself, not a selection filter -- a source's `patch-sets` list is the sole, exact determinant of what it applies. Whether a patch *applies* is what `patchset.py`'s anchors already verify on every build.

### Tree-state-key mechanics

A selection's effective tree state is a 16-character hex digest of the ref, the resolved `patch_set_id`, and the overlay digest (when the source has `overlay = true`). This fingerprint covers *only what changes the source tree* — builds, platforms, and variant-sets are cmake arguments and generated output, excluded deliberately so back-to-back builds don't flip the tree unnecessarily.

**Why it matters:** the 3-source default set (`llama-native` + `bigcherry-native` + `bigcherry`) resolves to only 2 distinct tree states: `llama-native` is unpatched, the other two apply the framework patch-set -- relevant to `apply`/`patches`, which still share one mutable checkout across sources. `build` (below) does not use this mechanism at all: each lane materialises its own isolated, content-addressed source, so there is no shared tree to reset.

### The bootstrap dependency chain

Applies to `[build.<name>]` entries in `config/recipes.toml`, selected via `build --lane SOURCE:BUILD:PLATFORM` or `--profile <name>` -- `needs` there is the literal, authoritative list; the summary below is illustrative, not exhaustive.

- **`stock`/`control`** — Just the dispatch layer, no tuning. Prerequisite for measuring (`record`).
- **`record`** — Measures signatures a real workload exercises. Produces `inventory.json` (via `bigcherry inventory record`).
- **`tune`** (`needs = ["inventory"]`) — Measures candidates against the workload. Produces `.measurements.jsonl`.
- **`replay`** (`needs = ["inventory", "promoted-winners"]`) — Compiles the replay layer. Loads winners from a promoted/exported dispatch cache (see [TEST.md](../testing/TEST.md)'s "Getting winners onto the hot path").

### Normal workflow

```bash
# 1. Verify the tree (audit + apply) -- shared-checkout patch selection
cd $BC/tools && python3 -m bigcherry audit
python3 -m bigcherry apply --source bigcherry

# 2. Build via the campaign engine (canonical v2 -- isolated per-lane sources)
python3 -m bigcherry build --lane bigcherry:record:linux-multi
# ...or a named profile from config/recipes.toml's [campaign.<name>]:
python3 -m bigcherry build --profile standard
# --all is shorthand for --profile standard

# 3. View what patches a source would use (apply/patches only, not build)
python3 -m bigcherry patches --source bigcherry
```

See `build --help` for the full current flag set (`--lane`/`--profile`/`--all`, `--inventory`/`--winners`, `--model`/`--hip-visible-devices` for real runtime-smoke validation, `--binary-relative-path` to select which binary a lane publishes as its primary artifact, e.g. `bin/llama-server` for a real server build vs the `bin/llama-bench` default). `--source`/`--variant-set`/`--force`/`--target` are NOT valid `build` flags -- argparse rejects them outright (exit 2); `--source` selects patches for `apply`/`patches` against the one shared checkout, a different axis from `build`'s isolated per-lane sources.

### Building for a specific GPU architecture

`--arch` overrides each lane's platform targets (it must be a non-empty
subset of them). `platform.linux-multi` declares all three cards, so a
single-architecture build is:

```bash
python3 -m bigcherry build --lane bigcherry:control:linux-multi --arch gfx1201
```

Each architecture gets its own `build_plan_id` and its own cached build tree,
so switching between them does not rebuild the others. Binaries land at:

```
~/.cache/bigcherry/builds/<source_slice_id>/<build_plan_id>/bin/
```

The `build` command prints `build_plan_id=` on success -- that is how you find
the binaries it just produced.

Note `build.control` produces `llama-bench` and the shared libraries but NOT
`llama-server`; a lane needing the server must say so via
`--binary-relative-path bin/llama-server`.

### The patch-qualification profile (4 arms)

`[campaign.standard]` varies the BUILD variant -- it is the autotune
record/tune/replay pipeline. `[campaign.patch-qualification]` varies the patch
COMPOSITION instead, which is the axis a patch is actually judged on:

| arm | source | carries the patch | answers |
|---|---|---|---|
| 1 | `llama-native` | no | what everything is ultimately measured against |
| 2 | `bigcherry-native` | no | control for the isolated A/B; vs arm 1, our framework's own cost |
| 3 | `bigcherry-native` | yes | the patch's ISOLATED effect |
| 4 | `bigcherry` | yes | the patch IN SITU, on top of everything already shipped |

Arm 4 exists because a patch worth +2% alone can be neutral or negative once
composed with the rest of the release set. Arms 3 and 4 answer different
questions and neither substitutes for the other.

```bash
python3 -m bigcherry build --profile patch-qualification --arch gfx1100
```

Arms 2 and 3 share one `source:build:platform` and differ ONLY by the
experiment. That is deliberate -- it is the pair that gives the isolated
comparison its meaning -- and it is why a campaign lane may declare its own
`experiment`:

```toml
{ source = "bigcherry-native", build = "control", platform = "linux-multi" },
{ source = "bigcherry-native", build = "control", platform = "linux-multi", experiment = "rd73-only" },
```

A request-level `--experiment` applies to EVERY lane, so it cannot express a
profile whose baselines must stay unpatched. A lane's own `experiment` wins
over the request-level one; lanes declaring none still inherit it. Lane ids
fold in the experiment, so the patched/unpatched pair does not collide with
the duplicate-lane check.

Swap the experiment name per patch under test; the rest of the profile is
invariant. An experiment name that does not exist is rejected at config load
rather than silently planning an arm identical to its baseline.

### Benchmarking a built lane

`llama-bench` has no `-c/--ctx-size`. Context depth is `-d/--n-depth`, which
prefills the KV cache to that depth before generating -- so a "48k context"
measurement is `-d 49152`:

```bash
B=~/.cache/bigcherry/builds/<slice>/<build_plan_id>/bin
export ROCR_VISIBLE_DEVICES=<card> HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$B
$B/llama-bench -m <model.gguf> -p 512 -n 128 -d 49152 -ctk q8_0 -ctv q8_0 -r 2
```

**Use identical settings on every card.** A 9B Q6_K is 7.7GB and 48k of f16 KV
is roughly 6.9GB, which is marginal in gfx1030's 16GiB; `q8_0` KV fits
everywhere with headroom. Tuning KV per card would make a cross-architecture
comparison meaningless, because KV quantisation changes the work being
measured. Fix one setting that fits the smallest card and use it on all of
them.

### Which build to measure on: the diagnostics split

A performance number and an activation proof must come from **two different
builds of the same revision**. Do not mix them.

| build | `GGML_HIP_DISPATCH_DIAGNOSTICS` | use it for |
|---|---|---|
| `replay` (what ships) | `OFF` (default) | **all performance numbers** |
| `replay-diagnostic`, or any `DIAGNOSTICS=ON` build | `ON` | activation evidence, hit rates, counters |

`GGML_HIP_DISPATCH_DIAGNOSTICS=OFF` is not "diagnostics quiet" -- the code is
not compiled at all. `dispatch_counters_enabled()` and
`native_select_timing_enabled()` return compile-time `false`, so every guarded
block is dead code the optimiser deletes, and the per-launch coverage counters
(two atomic RMWs per dispatch, ~382,000 in one measured bench run) are
`#ifdef`-ed out of their call sites. Setting a report path cannot switch them
back on.

`GGML_HIP_AUTOTUNE` or `GGML_HIP_AUTOTUNE_RECORD` imply diagnostics ON, because
tuning and recording need the counters to function. So a `tune` or `record`
lane is never a performance-measurement build.

Never use a timing from the diagnostics build as production performance, and never take activation
evidence from the production build -- the first is not what ships, and the
second cannot report anything.
Retain diagnostic observations with their own cell/build identity. They are
not same-cell production activation evidence. A linked companion protocol
must verify source/generated-input/cache/workload parity and explicitly
state the inference it supports; an arbitrary diagnostic run cannot admit
a production replay performance claim. See TEST.md for the remaining HI168
admission work.

### Device selection: the two-selector trap

`ROCR_VISIBLE_DEVICES` filters the device list FIRST, then
`HIP_VISIBLE_DEVICES` indexes **into that filtered list**. So setting both to
the same non-zero index selects *nothing*:

```bash
# WRONG -- asks for index 2 of a one-item list; selects no device
export ROCR_VISIBLE_DEVICES=2 HIP_VISIBLE_DEVICES=2

# RIGHT -- ROCR picks the card, HIP indexes within what survived
export ROCR_VISIBLE_DEVICES=2 HIP_VISIBLE_DEVICES=0
```

This is recorded as VA22 and it has since recurred, which is why it is
repeated here rather than left in a plan item.

**What makes it dangerous is the failure mode, not the mistake.** A ROCm init
failure does not stop llama.cpp: it falls back to CPU and still prints a
well-formed results table labelled backend `ROCm`. Measured on real hardware
2026-09-05, same binary, same model, same command:

```
failed to initialize ROCm: no ROCm-capable device is detected
| qwen35 9B Q6_K | ROCm | pp512 |   43.70 ± 0.06 |     <- CPU
| qwen35 9B Q6_K | ROCm | pp512 | 3160.50 ± 31.80 |     <- GPU
```

72x wrong, and the CPU run reported the *tighter* variance. The only signal
was one line of stderr above the table.

**Always confirm the positive init line before believing any number:**

```
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 32624 MiB):
  Device 0: AMD Radeon Graphics, gfx1201 (0x1201), ...
```

The campaign path enforces this automatically
(`experiment/attestation.py`, VA25); a hand-run `llama-bench` does not, so
check it yourself. Note llama-**server** does not emit that line at all --
even with `-v` -- so server lanes need different attestation (VA25 step 4).

### Card inventory (build server)

| device | arch | VRAM | notes |
|---|---|---|---|
| 0, 1 | gfx1100 | 24560 MiB each | 2x RX 7900 XTX; the only multi-GPU pair |
| 2 | gfx1201 | 32624 MiB | Radeon AI PRO R9700 |
| 3 | gfx1030 | 16368 MiB | RX 6900 XT; smallest, so it bounds model+KV choice |

gfx1030's 16 GiB is the binding constraint for any cross-architecture
comparison: a 27B Q8_0 (29GB) cannot run there at all, which is why
`tierB-qwen9b-q6k` (7.7GB) exists in `config/models.toml` as the model held
constant while architecture varies.

## Two gaps to know about

These apply to the **manual build cycle** below (standalone `generate` + raw cmake against `$BC`'s one shared checkout) -- `bigcherry build --lane`/`--profile` (the campaign engine) runs its own `generate` stage automatically per lane and does not have this gap.

1. **Manual `build` does not call `generate`** — it assumes the cmake options already exist. The first `generate --variant-set X` must run before `build` uses that variant-set, or cmake will reject the missing signature file. The separation lets you generate once (deterministic, ~1 min) and reuse across multiple compile variants.

2. **`replay-slim` needs `--winners`** — the slim catalog selects only the variants that a tuning run chose. You must run `generate --variant-set replay-slim --inventory ... --winners <measurements>` *before* building replay-slim, or the catalog will be empty and replay will fall back to native silently.

## Manual build cycle

One-off builds outside a source. `$BC` = `$BC_REPO` on the build server.

### Linux — all three GPUs

**Vendored ROCm toolchains.** The same `vendor/rocm/<version>/` convention
described in the Windows section below applies here — `tools/rocm-env.sh`
works unchanged on the build server. Two versions are already vendored under `$BC`:
`7.2.4` (copied from `/opt/rocm-7.2.4`, currently `/opt/rocm`'s target) and
`7.14` (copied from `/opt/rocm7140/rocm/core-7.14`, the version the older
manual-build snippets below reference). Select one before building:

```bash
cd $BC
source tools/rocm-env.sh --list
source tools/rocm-env.sh 7.2.4   # sets ROCM_PATH, HIP_PATH, prepends PATH
```

Then substitute `$HIP_PATH/llvm/bin/clang{,++}` for the hardcoded
`/opt/rocm/llvm/bin/clang{,++}` paths below if you want a non-default
version. System-wide `/opt/rocm*` still works unmodified if you don't
`source` a vendored version first.

```bash
cd $BC/tools
python3 -m bigcherry audit
python3 -m bigcherry apply --source bigcherry
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

**Vendored ROCm toolchains.** Multiple ROCm versions can live side-by-side
under `vendor/rocm/<version>/` (gitignored — never committed, ~3GB each).
Each is a straight copy of an AMD ROCm for Windows install tree, e.g.:

```powershell
robocopy 'C:\Program Files\AMD\ROCm\7.1' 'vendor\rocm\7.1' /E /MT:16
```

Select one for the current shell with `tools/rocm-env.ps1` (PowerShell) or
`tools/rocm-env.sh` (bash) — must be **dot-sourced/sourced**, not run, so the
env vars apply to your shell rather than a child process:

```powershell
. tools\rocm-env.ps1 -List     # see what's vendored
. tools\rocm-env.ps1 7.1       # sets ROCM_PATH, HIP_PATH, prepends PATH
```

```bash
source tools/rocm-env.sh --list
source tools/rocm-env.sh 7.1
```

This sets `ROCM_PATH`/`HIP_PATH` (the same variables
`tools/bigcherry/toolchain.py` captures for build-identity records), so any
recipe or manual `cmake` invocation below that references `$env:HIP_PATH`
picks up the selected vendored version. A system-wide ROCm install (e.g.
`C:\Program Files\AMD\ROCm\7.1`) still works unmodified if you'd rather not
vendor a copy — the two are interchangeable, just point `HIP_PATH` at
whichever tree you want.

```powershell
$env:PATH = 'C:\Program Files\AMD\ROCm\7.14\bin;' + $env:PATH
$env:HIP_PATH = 'C:\Program Files\AMD\ROCm\7.14'
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

### Building from git-bash on Windows

`tools/rocm-env.sh` sourced from git-bash can resolve `HIP_PATH` to a
backslash-containing Windows path (e.g. from a system-wide install ahead of
the vendored one) rather than the vendored `vendor/rocm/<version>` tree.
Passed straight into `cmake -D...`, a trailing-backslash-before-slash path
like `C:\Program Files\AMD\ROCm\7.1\/bin/...` can produce a genuinely broken
generated `CMakeRCCompiler.cmake` (`Invalid character escape '\P'`) rather
than a normal "file not found". Two independent gotchas worth knowing before
chasing a git-bash+cmake path error as something else:

1. **No resource compiler by default.** `CMAKE_RC_COMPILER` is unset unless
   you pass it; the vendored ROCm/LLVM toolchain bundles `llvm-rc.exe`, so
   `-DCMAKE_RC_COMPILER="$ROCM/bin/llvm-rc.exe"` (forward slashes) is the fix.
2. **Prefer forward-slash paths built from `pwd`** (e.g.
   `ROCM=$(pwd)/vendor/rocm/7.1`) over whatever `tools/rocm-env.sh` resolves
   `HIP_PATH` to, and set `HIP_PATH`/`ROCM_PATH`/`CMAKE_PREFIX_PATH` to that
   same value explicitly rather than trusting the sourced script's export —
   `find_package(hip)` needs `CMAKE_PREFIX_PATH` (or the `HIP_PATH`
   environment variable, not just a `-D` cache var) to locate
   `hip-config.cmake`.

Confirmed working end to end (2026-08-23, this exact machine): configure and
build `test-backend-ops`/`ggml-hip` against `vendor/rocm/7.1` and the local
AMD Radeon RX 7900 GRE (gfx1100), then run a real tuning sweep
(`GGML_HIP_DISPATCH_MODE=tune`) against it. Do not assume no compiler/GPU is
available in this environment without checking this section and
`vendor/rocm/` first.

## Known dependency/toolchain gaps

Anything vendored under `vendor/rocm/<version>/` is a straight copy of a
ROCm install and covers the compiler, HIP/hipBLAS/hipBLASLt *runtime*
libraries, and headers — everything a normal `GGML_HIP=ON` build needs. It
does **not** cover every ROCm-adjacent tool some tuning/experiment work
wants:

- **hipBLASLt offline-tuning client (`hipblaslt-bench`)** — not present in
  either the Windows HIP SDK vendored copy or the build server's `hipblaslt`/
  `hipblaslt-dev` apt packages (confirmed directly, 2026-08-23: the library
  and headers are there, the client binary is not, and there is no separate
  apt package for it). It has to be built from the `ROCm/hipBLASLt` GitHub
  source with `-DBUILD_CLIENTS=ON`, which additionally pulls in Tensile — a
  real, scoped, but substantially larger build than the library itself. See
  RD87 for the concrete next step when someone picks that up.

If you hit a similar "the runtime library is vendored but the CLI/dev tool
isn't" gap for some other ROCm component, add it to this list rather than
rediscovering it — the point of this section is to save the next person (or
session) the investigation.
