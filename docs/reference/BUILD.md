# Build reference

Environment setup and build commands for bigcherry recipes and manual configuration.

See also: [TEST.md](TEST.md) — testing, tuning workflows, and coverage.

## Environment — brutus (`10.10.100.10`)

**`~/bigcherry` is the live tree — a real local git clone.** Git operations
(fetch/checkout/status) over the old SMB-mounted path (below) were too slow
and unreliable over the network, so brutus builds from its own local
checkout now, not from the share. Treat it as a normal git worktree, not a
build cache: before building, `git status` must be clean and `git log -1`
must match what you expect (fetch/pull first if not); after any hands-on
edit there, commit and push before moving on.

**SMB share — file transfer only, never git/build:**

```text
J:\development\llmhosts\bigcherry  ==  /mnt/vault/development/llmhosts/bigcherry
```

Still useful for copying a build artifact or doc back and forth by hand.
Do not `git` anything against this path, and do not point cmake at it for
a brutus build — use `~/bigcherry` for that. (This section previously said
the reverse — that `~/bigcherry` was the stale copy and `/mnt/vault` was
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
  scp 10.10.100.10:/tmp/thing.md docs/reference/THING.md   # run from Windows
  ```

## Recipes — the normal way to build

`$BC` = `~/bigcherry` on brutus.

A **recipe** names one complete build configuration: an upstream ref, a patch selection, and which variant(s) to compile. Recipes live in `recipes.toml`.

### Recipe concept and axis table

| Axis | Meaning | Scope | Examples |
|------|---------|-------|----------|
| **Recipe** | One whole build identity | Global; names a row in recipes.toml | `upstream`, `bigcherry`, `release` |
| **Build** | A cmake variant set | Per-recipe; names one output tree | `record` (measures signatures), `tune` (tunes candidates), `replay` (applies winners) |
| **Platform** | GPU target(s) and compile flags | Per-recipe; single-GPU or multi-GPU | `linux-multi` (3 GPUs on brutus), `windows-gfx1100` (workstation) |
| **Patch state** | Patch acceptance status | Per-patch, orthogonal to ref | `validated`, `untested`, `rejected` |

Recipes intentionally avoid the name "profile" — three unrelated PROFILES already exist in this project (patchset groups, release_validate platforms, pareto_report objectives), and a fourth would make ambiguity permanent.

### Repinning

```bash
cd $BC/tools && python3 -m bigcherry repin [b<ref>]
```

Rewrites `recipes.toml`'s top-level `pinned = "..."` line in place, leaving comments intact. Recipes with `ref = "pinned"` (the default) now build from the new ref; recipes naming an explicit ref (e.g., `ref = "b10257"`) remain frozen and do not move. This lets you keep a historical recipe around for comparison without the file bloating.

### Patch state semantics

- **`validated`** — Measured, proven correct against the baseline (native).
- **`rejected`** — Measured or reviewed, rejected (too risky, no benefit, etc.).
- **`untested`** — New or awaiting measurement.

State is orthogonal to **group** (core/upstream-fixes) — state is a durable judgment about the change itself, not re-derived per pin. Whether a patch *applies* is what `patchset.py`'s anchors already verify on every build.

### Tree-state-key mechanics

A recipe's effective tree state is `tree_state_key(ref, groups, states)`, a 16-character hex digest of the ref, patch groups, and patch states. This fingerprint covers *only what changes the source tree* — builds, platforms, and variant-sets are cmake arguments and generated output, excluded deliberately so back-to-back builds don't flip the tree unnecessarily.

**Why it matters:** The 3-recipe default set (`upstream` + `bigcherry-native` + `bigcherry`) resolves to only 2 distinct tree states: upstream is unpatched, the other two select validated patches. Running `build --all` resets the tree *once*, not three times, saving ~25 minutes.

### The bootstrap dependency chain

- **`native`** — Just the dispatch layer, no tuning. Prerequisite for measuring (`record`).
- **`record`** — Measures signatures a real workload exercises. Produces `inventory.json`.
- **`tune`** (requires `needs = "inventory"`) — Measures candidates against the workload. Produces `.measurements.jsonl`.
- **`replay`** (requires `needs = "inventory"`) — Compiles the replay layer. Loads winners from measurements or exports them fresh to a cache.

### Normal workflow

```bash
# 1. Verify the tree (audit + apply)
cd $BC/tools && python3 -m bigcherry audit
python3 -m bigcherry apply

# 2. Build a single recipe (e.g., bigcherry)
python3 -m bigcherry build --recipe bigcherry

# 3. Or build all recipes with default=true
python3 -m bigcherry build --all

# 4. View what patches a recipe would use
python3 -m bigcherry patches --recipe release
```

## Two gaps to know about

1. **`build` does not call `generate`** — it assumes the cmake options already exist. The first `generate --variant-set X` must run before `build` uses that variant-set, or cmake will reject the missing signature file. The separation lets you generate once (deterministic, ~1 min) and reuse across multiple compile variants.

2. **`replay-slim` needs `--winners`** — the slim catalog selects only the variants that a tuning run chose. You must run `generate --variant-set replay-slim --inventory ... --winners <measurements>` *before* building replay-slim, or the catalog will be empty and replay will fall back to native silently.

## Manual build cycle

One-off builds outside a recipe. `$BC` = `~/bigcherry` on brutus.

### Linux — all three GPUs

**Vendored ROCm toolchains.** The same `vendor/rocm/<version>/` convention
described in the Windows section below applies here — `tools/rocm-env.sh`
works unchanged on Brutus. Two versions are already vendored under `$BC`:
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
