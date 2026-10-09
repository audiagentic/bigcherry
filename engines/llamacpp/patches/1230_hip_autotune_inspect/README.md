# 1230 (HI15/HI16): hip-autotune-inspect offline cache and registry inspector

## Scope

Wires a host executable (source in the overlay,
`hip-autotune-inspect.cpp`/`hip-autotune-replay.cpp`/`.h`) into the
`ggml-hip` backend build that links the same dispatch-candidate registry
functions and replay-cache loader a production process uses, plus a Python
`replay_inspect.py` wrapper -- so a Python reimplementation of the same
loading/checking logic can be verified against the real C++ implementation
rather than only against itself.

## Why

HI16's catalog/registry agreement tests and HI15's review both needed a
C++-side check that a Python reimplementation of the loader could not
itself introduce silent disagreement with the real registry.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework
(HI15/HI16).

## Real hardware evidence gathered so far (2026-08-22)

Two real Brutus/gfx1201 sessions (ledger events `chg_20260822_040544`,
`chg_20260822_044637`):

- Compiled clean (0 errors/warnings) in the real HIP device frontend on
  Brutus at gfx1201.
- `--selftest`: 10/10 known-answer cross-language digest vectors pass
  (C++ one-shot AND streaming blake2b vs Python hashlib, covering the three
  real runtime person prefixes, empty data, 128/129-byte block boundaries,
  multi-block, person zero-padding) -- caught and fixed a real
  byte-dropping bug in the tool's own streaming chunk-split on first run.
- Registry self-check: 88 candidates, 0 anomalies.
- Real dispatch cache (`dispatch-27b-v5.cache`) loads through the production
  loader: 59/59 winner slots usable.
- 17 offline Python wrapper tests pass; full offline suite passed at time of
  authoring.

## Lifecycle note -- deliberately still `untested`

Per the 2026-08-22 ledger record, this patch's `STATE` is deliberately kept
`"untested"` rather than promoted: under this project's HI83 contract,
`"validated"` requires a full HI82-style campaign record (the isolated,
content-addressed build/run pipeline used for real A/B patch campaigns), not
just a direct manual compile+link+selftest session, however thorough. That
campaign has not yet been run against this tool. This is not a gap this
README's authoring closes -- it documents the real, already-strong evidence
that exists while leaving the state field where the project's own policy
correctly leaves it.

## Known limitations

- No `validation.toml` adapter exists; this is a diagnostic/inspection tool
  patch (`kind = "diagnostic"`), not eligible for the
  `is_framework_configuration_patch` local-framework adapter path (which
  requires `kind = "framework"`).
- Promotion to `validated` requires the HI82 campaign record described
  above -- open follow-up work, not yet scheduled.
