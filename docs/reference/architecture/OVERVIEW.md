# HIP Autotune — Plan Overview

> This page preserves an historical architecture/phase snapshot. Current
> implementation status and completion state are authoritative only in the
> corresponding plan items under `docs/planning/{active,completed}/`.

Derived from `llama_hip_autotune_prework` design and source-audited
implementation material. See [PACK_REVIEW.md](../archive/PACK_REVIEW.md) for the delta
between that pack, these plans, and what building against a real checkout revealed.

**Original audit target:** `ggml-org/llama.cpp` at `0ef6e55edb306fcbcf73e6f1f41923cccb9cf7f8`
**Historical snapshot revision:** `22dc605` — all 32 audited invariants held at
the time of this snapshot
**Targets:** every AMD GPU llama.cpp supports — 26 architectures, GCN4 through RDNA4

Delivered as a release-tolerant **overlay** (`src/` + anchored `patches/`), not
a fork, so each new upstream release is re-audited, re-patched and rebuilt
rather than merged. See the repository README.

## Phase 0 — Setup and baseline

- **HI01** — Pin upstream, run source audit, establish baseline builds (M) — **done**

## Phase 1 — Infrastructure

- **HI02** — CMake options, ABI types, HIP autotune header infrastructure (M) — **done**
- **HI03** — Candidate catalog generator — single source of truth (L) — **done**
- **HI04** — Separate native selection from launch — no behavior change (L) — **done, exercised on hardware**
- **HI05** — Canonical signature construction and hardware key (M) — **done (blake2b verified)**

## Phase 2 — Expose existing choices

- **HI06** — MMQ forced-J variant enumeration and dispatch (M) — **done, verified**
- **HI07** — MMVF forced block-size variant dispatch (M) — **done, verified**
- **HI08** — MMF forced-nwarps variant dispatch (M) — **done, verified**

## Phase 3 — Generate new variants

- **HI09** — Explicit MMVQ geometry variants (L) — **done**; `small_k` added, fusion correctly excluded (PACK_REVIEW B1)
- **HI09b** — Parse compiler resource reports, blacklist unviable geometries (M) — *not started*

## Phase 4 — Record, replay, and tune

- **HI10** — Signature collection (record mode) and SQLite persistence (L) — **done** — JSONL + offline SQLite
- **HI11** — Compact replay cache and manual seeding before tuning (M) — **done**, loader and writer
- **HI12** — Family-level tuning engine with complete-path timing (L) — **done**, sweep completes end to end

## Phase 5 — Fused paths and advanced

- **HI13** — Fused path collection and MUL_MAT_ID dispatch (M) — **done**; promoted ahead of HI12, coverage complete

## Phase 6 — Validation and production

- **HI14** — Multi-GPU validation and graph capture verification (L) — *partly done*
- **HI15** — Production hardening — replay-full binary, cache export (L) — *mostly done*
- **HI16** — Comprehensive test suite (L) — *partial*

## Phase 7 — Taxonomy and the remaining opaque paths

Added after auditing what the tuner can actually choose between.
See [FAMILY_MODEL.md](FAMILY_MODEL.md) for source-level verification.

- **HI19** — Enforce the signature / context / candidate / observation separation (M) — *not started*; do first
- **HI17** — Decompose the opaque BLAS candidate: conversion routes, `compute_type`,
  output conversion, `api_strategy`, provider policies (L). See coverage audit for impact.
- **HI18** — Tune the multi-device allreduce (`nccl` / `internal` / `butterfly`) (M). See coverage audit.

Rejected: WMMA as a top-level family. See [FAMILY_MODEL.md](FAMILY_MODEL.md).

## Key principles

1. **No behavior change until proven** — native parity is required at every refactor step
2. **Complete-path timing** — measure the full candidate path including setup/quantization
3. **One source of truth** — catalog generator produces all artifacts, not maintained independently
4. **Production never benchmarks** — replay builds contain no tuning engine or SQLite
5. **Hard eligibility before launch** — reject invalid candidates at compile or selection time, not runtime
6. **Native and forced share one launcher.** Every forced-variant refactor lifts the existing switch into a launcher taking the value as an argument, then reduces the native path to computing its value and calling the same launcher. Forced-native is identical to native *by construction*, not by testing.
7. **Derive from upstream, never restate it.** The MMQ config tables are sparse and change between releases; the catalog parses them. Anything restated in bigcherry is a copy that can silently disagree with the tree it patches.
8. **A cross-language identity must be pinned by test.** Dispatch keys are computed in C++ at runtime and Python offline. A divergence does not fail loudly — every lookup simply misses and falls back to native, and the system appears to work while doing nothing.

## Verification summary

| Area | Status |
| --- | --- |
| Source audit | 32/32 checks, on pristine and patched trees |
| CMake validation | all 5 rejection rules, by real `cmake` runs |
| Catalog | 3062 candidates / 26 architectures; hash deterministic and order-independent |
| blake2b | 7/7 vectors match Python `hashlib`, incl. RFC 7693 |
| Patch engine | 10/10 regression tests |
| Dispatch layer (end to end) | signature → blake2b → dispatch-key → cache → replay → native fallback → miss recording, all on real hardware |
