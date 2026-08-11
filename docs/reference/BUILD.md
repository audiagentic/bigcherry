# Build reference

Environment setup and build commands for bigcherry recipes and manual configuration.

See also: [TEST.md](TEST.md) — testing, tuning workflows, and coverage.

## Environment — brutus (`10.10.100.10`)

**SMB share mapping:**

```text
J:\development\llmhosts\bigcherry  ==  /mnt/vault/development/llmhosts/bigcherry
```

No copy or sync step needed — edit on either side, build on the server.

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

- **`~/bigcherry` on brutus is a stale copy** — ignore or delete it. The live tree is under `/mnt/vault`.

## Recipes — the normal way to build

`$BC` = `/mnt/vault/development/llmhosts/bigcherry` (or `J:\development\llmhosts\bigcherry`).

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

One-off builds outside a recipe. `$BC` = `/mnt/vault/development/llmhosts/bigcherry` (or `J:\development\llmhosts\bigcherry`).

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
