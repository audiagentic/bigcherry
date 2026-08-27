# BigCherry tools

Normative map: [`docs/reference/tooling/TOOLING.md`](../docs/reference/tooling/TOOLING.md).

Canonical maintained domains are under `tools/bigcherry/`: `release`, `source`,
`build`, `experiment`, `patch`, `campaign`, `tuning`, `core`, and `analysis`.
Use `bigcherry.cli` for command parsing/presentation and retain
`bigcherry.__main__` only as the supported entrypoint compatibility surface.

- Release: `repin`, `pin-status`, `pull`, `audit`, release validation/records
- Source/build: `audit`, `generate`, `build`
- Patches: `patches`, `patch-status`, `patch-explain`, `patch-lint`, `patch-validate`
- Campaigns: canonical planner/lane/build/smoke/compare/benchmark paths
- Tuning/replay: catalog, journal, promotion, correctness, ranking, replay
- Experiments: `experiment-contract` and bundles
- Diagnostics: `check` and `doctor` (deterministic, non-mutating, hardware-free)
- Exploration: `tools/lab/<plan-topic>`; not a package or evidence authority
- Shell setup: `tools/env/`

Root modules that still forward to canonical domains are compatibility facades,
not new implementation destinations. Remove one only with consumer, identity,
and parity evidence recorded by the rationalisation work.

Permanent tests remain under `tools/tests`; only the release slice is currently
under `tools/tests/release/`. Search before creating a second framework. Record
substantive changes through the planning and ledger MCP processes, and preserve
shared-worktree changes belonging to other actors.
