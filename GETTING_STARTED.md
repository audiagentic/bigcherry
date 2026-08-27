# Getting started (agents / new contexts)

Task-specific context (which plan item, which patch, which experiment) lives in
`docs/planning/`. This file is the opposite: how to build, test, and run
things in this repo regardless of what task brought you here. See `README.md`
for the architecture (src/patches overlay model, MMQ catalog design).

## Repo vs. real hardware

- This checkout builds and runs on CPU-only dev machines too, but the AMD
  ROCm/HIP GPUs this project targets live on a remote host. Real
  measurement, tuning, and correctness work happens there over SSH — a local
  `pytest` pass proves the Python tooling is correct, not that a kernel is
  fast or correct on real silicon.
- Never report a benchmark/tuning number you didn't personally observe from a
  real run. If you can't reach the GPU host, say so — don't estimate.
- `rocm-smi --showmeminfo vram --showpids` before touching any GPU process —
  don't assume the hardware is free, and don't kill a live process without
  explicit authorization.

## Build/test loop (Python tooling, no GPU needed)

```bash
cd tools && python -m pytest ../tools/tests -q      # or: pytest (uses pytest.ini)
PYTHONPATH=tools python -m bigcherry doctor          # environment sanity check
PYTHONPATH=tools python -m bigcherry check --default # deterministic local CI gates (--quick/--full also exist)
```

## Taking a new upstream llama.cpp release (or standing up a fresh checkout)

```bash
PYTHONPATH=tools python -m bigcherry pull   --ref <upstream-rev>
PYTHONPATH=tools python -m bigcherry patch-rebase-check --recipe bigcherry --json releases/patch-rebase.json
PYTHONPATH=tools python -m bigcherry apply --rebase-report releases/patch-rebase.json --known-good  # or: apply (all-or-nothing)
PYTHONPATH=tools python -m bigcherry audit                # strict invariant audit — must pass before generate
PYTHONPATH=tools python -m bigcherry generate --arch all   # candidate catalog -> artifacts
```

Each stage refuses to run on a tree that hasn't passed the stage before it.
`patch-rebase-check` is the exception: it is observational (an isolated
detached-worktree probe of whether the selected patches' anchors still find
their targets in the revision `pull` just moved to) and never advances
release stage on its own. It reports each patch as `CLEAN`/`CLEAN_NOOP`/
`NOT_APPLICABLE_BY_DESIGN` (fine) or `FAILED_NEEDS_RECONCILIATION`/
`BLOCKED_BY_DEPENDENCY`/`QUARANTINED` (needs a human) — see
`docs/reference/PIN_BUMP.md` for the full bump runbook, including how to
apply just the known-good subset while the rest gets fixed. A plain `apply`
(no `--rebase-report`) is unchanged: still all-or-nothing, still fails
closed on the first anchor that doesn't find its target.

## Building a binary

```bash
PYTHONPATH=tools python -m bigcherry build \
  --lane <source>:<build>:<platform> \
  --binary-relative-path bin/llama-server \
  [--inventory PATH] [--winners PATH] [--experiment NAME] [--run-id ID]
```

Real `source` names: `bigcherry` (normal patch-set), `bigcherry-native`
(framework only, no validated-enhancement patches — the fair "before" baseline
for an A/B), `llama-native` (genuinely stock upstream, same pin, zero
patches). Real `build` names: `stock`, `control`, `record`, `tune`, `replay`,
`audit`. Real `platform`: `linux-multi`, `windows-gfx1100`, `vulkan-linux`.
`--inventory`/`--winners` are required by builds whose recipe declares
`needs = [...]` for them (`record`/`tune` need inventory; `replay` needs
both). Full lane/recipe list: `config/recipes.toml`.

## Patches — inspecting and validating one

```bash
PYTHONPATH=tools python -m bigcherry patches                 # list with group/state/upstream status
PYTHONPATH=tools python -m bigcherry patch-explain <name>     # source, requires/conflicts, which recipes select it
PYTHONPATH=tools python -m bigcherry patch-lint                # metadata lint, no mutation
PYTHONPATH=tools python -m bigcherry patch-validate <name>    # run its validation.toml checks
```

A patch is an anchored, regex-located edit to an upstream-owned file (see
`patches/<name>/`) — it fails loudly, naming the missing anchor, rather than
silently mis-applying when upstream shifts underneath it. Whole new files go
in `src/` instead and never conflict. New patch scaffold: `patches/_template/`.

## Runtime dispatch modes (what a built binary actually does)

| `GGML_HIP_DISPATCH_MODE` | Needs | Behavior |
| --- | --- | --- |
| (unset / native) | — | normal upstream dispatch |
| `record` | `GGML_HIP_DISPATCH_DB=<path>` | logs real dispatch signatures seen |
| `tune` | `GGML_HIP_DISPATCH_DB=<path>` | benchmarks candidates live against real traffic |
| `replay` | `GGML_HIP_DISPATCH_CACHE=<path>` | dispatches promoted winners, falls back to native on a miss |

`record`/`tune`/`replay` are **mutually exclusive compile-time CMake
options** — one binary is always exactly one mode; there's no runtime switch
between them.

## Gotchas (learned the hard way — real hardware findings)

- **Build artifact collisions (HI110, open):** `build_plan_id` is
  content-addressed from source+patches+config only — it does **not**
  account for the requested binary target. Building a second binary (e.g.
  `test-backend-ops`) into a directory that already published a first one
  (e.g. `llama-server`) under the identical lane/experiment/inventory can
  fail with `ArtifactError: immutable artifact already exists with
  different bytes`, even though the actual compile succeeds. Workaround:
  the binary is usually still sitting compiled at
  `<build_dir>/bin/<target>` even though publish failed — use it directly.
  A distinct `--experiment` also forces a fresh `build_plan_id` if you need
  a clean publish.
- **Correctness-evidence binary scoping (HI106):** when generating
  correctness evidence (`bigcherry.hi80_generate_correctness_evidence`),
  the `test-backend-ops` binary passed via `--binary` **must** be built from
  the exact same `--inventory` as the tune run being evidenced. A binary
  built from a different/narrower inventory aborts with
  `GGML_HIP_FORCE_CANDIDATE=... not found in registry` — a tooling-scope
  error, not a real correctness bug (the error message says so explicitly).
- **Tune mode needs extra VRAM headroom.** Live candidate benchmarking
  allocates scratch workspace on top of whatever the model/KV-cache already
  use — production KV-cache settings that fit fine in `native`/`replay` can
  OOM under `GGML_HIP_DISPATCH_MODE=tune`. Reduce `--parallel`/`-c` for the
  tuning run if you hit a ROCm OOM right after model load.
- **A long-running `llama-server` only flushes `GGML_HIP_DISPATCH_DB` data
  on clean exit.** It never exits on its own. Use the opt-in
  `POST /shutdown` endpoint (`LLAMA_SERVER_ENABLE_SHUTDOWN=1`) instead of
  killing the process, or record/tune data is silently lost.
- **Quantized types (Q8_0/Q4_K/Q6_K/...) are supported in correctness
  evidence as of HI111** — pass `nb=0` sentinel strides to test-backend-ops
  rather than hand-computing byte strides; don't reintroduce a
  float-only type whitelist.

## Source control

Use the MCP git tools, not raw `git`/`gh`, per `AGENTS.md`. This is a shared
multi-agent working tree: never `git stash`/`reset --hard`/force-push, and
never commit a file you didn't actually change (check `git status`/`git
diff` before staging — other sessions leave in-progress work uncommitted
here routinely).

## Process doctrine (ledger + planning)

See `AGENTS.md` / `CLAUDE.md` — record a ledger change event after
substantive implementation work, manage multi-step work as plan items under
`docs/planning/` via the `ag-planning` MCP tools.
