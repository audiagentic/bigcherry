# RCS11: pin ROCm compiler and runtime to the selected SDK

Archived from `rcs11-merge` (tip `76842d316`, original common ancestor `a40258234` before the history rewrite). Branch-specific work dates to 2026-11-02.

## What and why

The build previously could select a compiler from one ROCm SDK but discover HIP/HSA libraries from another. The branch added `engine.toml` `[build].rocm-paths`, strict `selected_rocm_root()` resolution and toolchain/build identity from the resolved SDK; it documented verification and the handoff to BRVP04.

## Outcome and why stopped

**(a): two change groups (SDK pinning and related planning/ledger/tests) already shipped through PR #524; (b): none; (c): none.** Main commit `13ab746ca` contains the same selected SDK design, with `rocm-paths` and `selected_rocm_root()` still present. The branch was made redundant by the merged PR. No alternative unpublished implementation needs a patch.

The RCS11 plan records a successful container configure/link, pinned compiler/runtime paths in component CMakeCache, negative SDK-root tests and unchanged llama.cpp behavior. No throughput measurement was needed for this build correctness fix.

## Evidence and resumption

- Branch commit `4240880fc` (implementation), `58ed6ab20` (validation and ledger), `937a25627` (completed plan and BRVP04 handoff); main PR #524, commit `13ab746ca`.
- `docs/planning/completed/build-radiance-components/RCS11.md`; `docs/planning/active/build-rocm-vulkan-provider/BRVP04.md`; `docs/releases/CURRENT_RELEASE_LEDGER.ndjson`.
- Resume from current selected-SDK code. For per-lane stack selection follow BRVP04; do not reintroduce fallback SDK discovery or historical absolute paths.
