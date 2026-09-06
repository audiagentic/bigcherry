# Environment reference

Machine-specific facts live in [`config/environment.toml`](../../config/environment.toml).
Documentation names a **role**; the config resolves it to a host.

## Why

The reference docs carried 83 occurrences of a hostname, 64 of a share path,
39 of a home directory and 15 hardcoded IP addresses across 40 files. That
makes the documentation describe one person's machine rather than the
procedure, and it turns "use a second build host" into a 40-file edit instead
of a config change.

## Roles

| role | what it is |
|---|---|
| **build server** | the multi-GPU AMD host that owns every real GPU measurement: builds, tune campaigns, profiling, end-to-end benchmarks |

Prose should say "the build server". Commands should use `$BC_HOST` or the
ssh alias, never a literal address.

## Using it

```bash
source tools/bigcherry-env.sh              # default host
source tools/bigcherry-env.sh build-server # a named host
```

Exports:

| variable | meaning |
|---|---|
| `BC_HOST` / `BC_ADDRESS` | ssh alias / IP (prefer the alias) |
| `BC_REPO` | the live git checkout builds run from |
| `BC_CACHE` | content-addressed build and campaign cache |
| `BC_MODEL_ROOT` | gguf model root |
| `BC_BENCH_HARNESS` | the documented `run_bench.py` server-bench harness |
| `BC_SHARE` | SMB share — file transfer only, never git or a build target |
| `BC_ROCM` / `BC_ROCM_SHIM` | ROCm install / the shim campaigns require |
| `BC_BENCH_PORT` / `BC_PRODUCTION_PORT` | benchmark port / production service port |
| `BC_DEVICES` | `index:arch` pairs, e.g. `0:gfx1100 1:gfx1100 2:gfx1201 3:gfx1030` |

On the host it also exports `ROCM_PATH`, `HIP_PATH` and prepends the shim to
`PATH`, because campaigns build compiler paths as `<hip-path>/bin/clang` and
`/opt/rocm` ships `amdclang`/`amdclang++` with real `clang` hidden under
`llvm/bin`.

So instead of:

```bash
ssh brutus 'cd /mnt/vault/development/llmhosts/llamacpp && python3 bench/run_bench.py ...'
```

write:

```bash
source tools/bigcherry-env.sh
ssh "$BC_HOST" "cd $BC_BENCH_HARNESS/.. && python3 bench/run_bench.py ..."
```

## Two facts worth keeping in the config, not in prose

**Ports.** `BC_PRODUCTION_PORT` (8080) belongs to the production inference
service. A benchmark that binds it takes production down; a tuner that assumes
it will fail to start. This is not hypothetical — a tune campaign died at its
first stage on exactly that collision. Ad-hoc servers should take a free port.

**Devices.** The index is the HIP/ROCR visible-devices ordinal that `--devices`
and `*_VISIBLE_DEVICES` take. Two `gfx1100` cards at index 0 and 1, one
`gfx1201`, one `gfx1030` — and the smallest card's VRAM is what constrains any
cross-architecture comparison, because changing model or KV quantisation per
card changes the work being measured.

## Not the same as `[[trees]]`

`config/recipes.toml`'s `[[trees]]` models repo **checkouts** — name, path,
required, role, expected tooling revision. It carries no address, model root,
toolchain or device inventory. The two are orthogonal: one host can hold
several trees, and a tree's path means nothing without the host it lives on.

## Migration status

`config/environment.toml` and `tools/bigcherry-env.sh` exist and resolve
correctly. The existing docs have **not** yet been rewritten to use roles —
that is a mechanical pass over ~40 files and is tracked separately. Until it
lands, treat this file as the source of truth for environment facts and the
older docs' literal paths as illustrative of one host.
