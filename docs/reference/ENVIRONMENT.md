# Environment reference

Machine-specific facts live in `config/environment.local.toml` (gitignored; created from
[`config/environment.example.toml`](../../config/environment.example.toml)).
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
ssh alias, never a literal address. The host is named by its key in the config
(`default-host`, `[host.<name>]`); a second host is another `[host.<name>]` table.

## Using it

```bash
source tools/env/bigcherry-env.sh              # default host
source tools/env/bigcherry-env.sh dev          # a named host
```

Exports:

| variable | meaning |
|---|---|
| `BC_HOST` | ssh alias (prefer the alias to any address) |
| `BC_REPO_ROOT` | the live git checkout builds run from; also the default base for relative SDK roots |
| `BC_CACHE_ROOT` | content-addressed build and campaign cache |
| `BC_MODEL_ROOT` | gguf model root |
| `BC_BENCH_HARNESS` | the documented `run_bench.py` server-bench harness |
| `BC_ROCM_ROOT` | the default ROCm install (`rocm-root`) |
| `BC_TEST_PORT` | port for tests, benchmarks and ad-hoc servers |
| `BC_PRODUCTION_PORT` | the production service port (must never be used by a test) |
| `BC_DEVICES` | `index:arch` pairs, e.g. `0:gfx1100 1:gfx1100 2:gfx1201 3:gfx1030` |

Any of the `BC_*_ROOT` variables can be overridden by setting it in the process
environment or in the `[env]` table of the local config; otherwise it comes from
the host's keys above.

## ROCm

`rocm-root` is the default ROCm install. Other SDKs are named in the host's
`rocm-sdks` table, mapping a version name to a root:

```toml
[host.dev]
rocm-root = "/opt/rocm"

[host.dev.rocm-sdks]
"7.2"  = "/opt/rocm-7.2.4"
"10.1" = "vendor/rocm/10.1.0rc2"   # a relative root is resolved against repo-root
```

Recipe stacks name a version, never a path: `sdk = "7.2"` in
`config/recipes.toml`. Adding or removing an SDK is an edit to the `rocm-sdks`
table; no recipe changes.

The real clang driver lives under `<root>/llvm/bin`. `/opt/rocm` ships
`amdclang` and `amdclang++` there, and the compile check looks for `llvm/bin`
first. No shim is needed.

## Two facts worth keeping in the config, not in prose

**Ports.** `BC_PRODUCTION_PORT` (8080) belongs to the production inference
service. A benchmark that binds it takes production down; a tuner that assumes
it will fail to start. This is not hypothetical — a tune campaign died at its
first stage on exactly that collision. The engine bench takes its port from
`test-port` unless `--port` is given, and it refuses the production port.

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
The `dev` tree's path is the host's `BC_REPO_ROOT`.

## Migration status

`docs/reference` contains no hostname, IP address, home directory, share path,
or superseded variable name (`BC_REPO`, `BC_ROCM`, `BC_CACHE`, `BC_SHARE`,
`BC_HOME`, `BC_ADDRESS`, `BC_BENCH_PORT`, `BC_ROCM_SHIM`, `BIGCHERRY_ROCM_*`,
`sdk-root`). Verified by grep. Prose names the role; commands use `$BC_*`.

Historical records in `docs/planning`, `docs/archive` and `docs/releases` describe
what was done at the time. Host facts in them have been replaced with the same
placeholders, and the scanner (`python3 -m bigcherry.forbidden_terms --all`)
reports none in `docs/`.
