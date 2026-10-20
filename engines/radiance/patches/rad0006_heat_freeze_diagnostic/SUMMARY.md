# rad0006_heat_freeze_diagnostic

**Status:** untested
**Plan item:** RAD14

## What it does

Adds `RAD_DIAG_FREEZE_PROMOTION_AFTER=N` to the heat engine: from its (N+1)th `HeatEngine::step()` call on, no swaps are proposed, so the resident expert set stops changing while the engine keeps serving (unlike `--deterministic`, which pins the loader's plan and did not fit the R9700 for the 35B). Unset or 0 changes nothing. One log line says when the freeze starts.

Diagnostic builds only: the code sits inside `#ifdef BC_MOE_REMOTE_DIAG` (`diagnostic-build` in `patch.toml`), and patch-lint keeps the package out of any source or patch set not named `*-diag`. It is in the `moe-remote-diag` set so the existing `radiance-diag` source and `build.moe-diag` lane carry it.

## Why

Rule out the expert mover as the cause of the Qwen3.6-35B-A3B MTP acceptance collapse (RAD14). Result: freezing promotions after 400 steps did not restore acceptance, so the mover is not the cause; the trigger is the prefix cache (see `docs/reference/engines/radiance/models/qwen3.6-35b-a3b.md`).

## Evidence

Built on Brutus (`radiance-diag:moe-diag:linux-radiance`, plan 058e639b747b85e5177bdab8b38f6ba3) and run once (`qwen35b-engine-mtp3-freeze.json`): the server log printed "promotions frozen after 400 steps" and acceptance stayed at 0.02-0.07 at 4 and 6 streams. Kept as `untested`, a lab diagnostic outside the production set.
