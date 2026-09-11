# 1232 (HI81): route CMAKE_HIP_FLAGS-gated options through the real CXX compile rule on Windows

## Scope

On Windows, `CXX_IS_HIPCC` is forced `TRUE` unconditionally in
`ggml/src/ggml-hip/CMakeLists.txt` (CMake's HIP language is never enabled
there), so ROCm source files compile as plain CXX, not through CMake's HIP
language rule -- meaning anything appended to `CMAKE_HIP_FLAGS` (the
pre-existing `GGML_HIP_EXPORT_METRICS` option and this project's own
`GGML_HIP_UNSAFE_MATH`, patch 1002) is a silent no-op there: the CMake cache
variable is set and looks correct, but the actual CXX compile rule never
consults it. Under the `CXX_IS_HIPCC` branch, this patch appends the same
flags to the ROCm sources' real CXX `COMPILE_FLAGS` via
`set_property(... APPEND_STRING ...)`, composing with (not clobbering) the
pre-existing Windows-Debug `-O2 -g` workaround at the same anchor site.

## Why

Root-caused by reading the CMake source directly (not guessed): confirmed
via `CMakeCache.txt` showing the flag as set while `ninja -t commands` never
showed it reaching the actual compile line, for both `CMAKE_HIP_FLAGS`
consumers in this file.

## Upstream / provenance

Local design -- a real build-configuration bug found and fixed in this
project (HI81), not ported from any external source.

## Real hardware evidence (2026-08-23)

Real-hardware validated on Windows, gfx1100 (RX 7900 GRE), ROCm 7.1, Ninja
(ledger event `chg_20260823_071308`):

- `-funsafe-math-optimizations` correctly appears in the real
  `ninja -t commands` HIP compile line when `GGML_HIP_UNSAFE_MATH=ON`.
- Correctly absent when `GGML_HIP_UNSAFE_MATH=OFF` (default).
- `ggml-hip.dll` builds clean in both configurations.
- Root cause independently corroborated by an external GPT agent
  (`dev-gpt-agent`, via the agents gateway, `req_8acb7c7f51384d64`), reached
  from first principles without repo access -- confirms the mechanism is not
  an artifact of this project's own framing.

This is a deterministic compile-command check (exact string match in the
real compiled command line), not a statistical benchmark comparison --
stronger signal than a timing-based claim would be for a build-configuration
correctness fix.

## Lifecycle note

`patch.py`'s own `STATE` constant and `patch.toml`'s `state` field both said
`"untested"` with an inline comment noting the patch was written and its
mechanism confirmed by source-reading alone, pending real Windows hardware
validation. That validation was completed in the same 2026-08-23 session
(see ledger event above) but the state fields were never updated to match --
corrected to `"validated"` here as a metadata-sync fix, not a new promotion
decision. `validation-architectures` backfilled to `["gfx1100"]`, the one
architecture this patch has actually been validated against (Windows-only
fix; inert on Linux by design, where `CXX_IS_HIPCC` is false and the
existing `CMAKE_HIP_FLAGS` path already worked).

## Known limitations

- No `validation.toml` adapter exists. This is a CMake build-flag-routing
  fix with no runtime code path to trace-mark; its evidence is the
  deterministic compile-command check above, produced by manual
  investigation rather than the generic campaign harness.
- Only validated on gfx1100/Windows -- the one platform this patch actually
  changes behavior on. Not re-validated on other Windows GPU architectures.
