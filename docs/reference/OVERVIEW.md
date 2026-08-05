# HIP Autotune — Plan Overview

Derived from `llama_hip_autotune_prework` design and source-audited
implementation material. See `PACK_REVIEW.md` for the delta between that pack,
these plans, and what building against a real checkout revealed.

**Original audit target:** `ggml-org/llama.cpp` at `0ef6e55edb306fcbcf73e6f1f41923cccb9cf7f8`
**Current working revision:** `22dc605` — all 32 audited invariants still hold
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

- **HI06** — MMQ forced-J variant enumeration and dispatch (M) — **done, verified on gfx1100**
- **HI07** — MMVF forced block-size variant dispatch (M) — **done, verified**
- **HI08** — MMF forced-nwarps variant dispatch (M) — **done, verified**

## Phase 3 — Generate new variants

- **HI09** — Explicit MMVQ geometry variants (L) — **done**; `small_k` added, fusion correctly excluded (PACK_REVIEW B1)
- **HI09b** — Parse compiler resource reports, blacklist unviable geometries (M) — *new, from plan 17.2*

## Phase 4 — Record, replay, and tune

- **HI10** — Signature collection (record mode) and SQLite persistence (L)
- **HI11** — Compact replay cache and manual seeding before tuning (M)
- **HI12** — Family-level tuning engine with complete-path timing (L)

## Phase 5 — Fused paths and advanced

- **HI13** — Fused path collection and MUL_MAT_ID dispatch (M) — **promoted: should precede HI12**

  Measured on gfx1100 with Qwen3.5-2B: token generation is only **66.9%**
  covered by measured dispatch, and the entire deficit is MMVQ (292/508 = 57.5%)
  reaching the kernel through fused graph paths that bypass the dense selector.
  Prompt processing is 100%. Tuning before this lands would optimise a
  known-incomplete population of the workload that matters most for latency.

## Phase 6 — Validation and production

- **HI14** — Multi-GPU validation and graph capture verification (L)
- **HI15** — Production hardening — replay-full binary, cache export (L)
- **HI16** — Comprehensive test suite (L)

## Phase 7 — Taxonomy and the remaining opaque paths

Added 2026-08-05 after auditing what the tuner can actually choose between.
See [FAMILY_MODEL.md](FAMILY_MODEL.md) for the source-level verification.

- **HI19** — Enforce the signature / context / candidate / observation
  separation (M) — **do first**; retrofitting it onto two new families costs
  more than establishing it up front
- **HI17** — Decompose the opaque BLAS candidate: conversion routes,
  `compute_type`, output conversion, `api_strategy`, provider policies (L).
  19 of 92 signatures on the production profile resolve to BLAS and can
  currently only pick "whatever hipBLAS decides"
- **HI18** — Tune the multi-device allreduce (`nccl` / `internal` / `butterfly`)
  (M). On the path of every tensor-split matmul on the 2× XTX rig, chosen once
  at backend construction and never measured

Rejected: WMMA as a top-level family. `rocwmma` appears nowhere in the ggml
sources; `amd_wmma_available` selects code paths *inside* MMQ, MMF and
FlashAttention. Promoting it would create a family with no entry point.

## Key principles

1. **No behavior change until proven** — native parity is required at every refactor step
2. **Complete-path timing** — measure the full candidate path including setup/quantization
3. **One source of truth** — catalog generator produces all artifacts, not maintained independently
4. **Production never benchmarks** — replay builds contain no tuning engine or SQLite
5. **Hard eligibility before launch** — reject invalid candidates at compile or selection time, not runtime

## Principles added during implementation

6. **Native and forced share one launcher.** Every forced-variant refactor
   lifts the existing switch into a launcher taking the value as an argument,
   then reduces the native path to computing its value and calling the same
   launcher. Forced-native is then identical to native *by construction*, not
   by testing. A parallel forced path would be free to drift; this cannot.
7. **Derive from upstream, never restate it.** The MMQ config tables are sparse
   and change between releases; the catalog parses them. Anything restated in
   bigcherry is a copy that can silently disagree with the tree it patches.
8. **A cross-language identity must be pinned by test.** Dispatch keys are
   computed in C++ at runtime and Python offline. A divergence does not fail
   loudly — every lookup simply misses and falls back to native, and the system
   appears to work while doing nothing.

## Status

| Verified | Evidence |
| --- | --- |
| Source audit | 32/32 checks, on pristine and patched trees |
| CMake validation | all 5 rejection rules, by real `cmake` runs |
| Catalog | 3062 candidates / 26 architectures; hash deterministic and order-independent |
| blake2b | 7/7 vectors match Python `hashlib`, incl. RFC 7693 |
| Patch engine | 10/10 regression tests |
| gfx1100 build | clean, with HI06 applied |
| Correctness (native) | 1545/1545 `test-backend-ops` MUL_MAT cases |
| Correctness (replay) | 1545/1545, 1151 dispatch keys recorded |
| Signature hygiene | 0 diagnostic fields leaked; 1151/1151 canonically sorted |
| Corrupt-cache fallback | verified on hardware — safe native fallback, no failures |
| MMVQ geometry | 56 explicit instances compile; native instantiation unchanged |
| Dispatch coverage | pp 100%, **tg 66.9%** — see HI13 |

The dispatch layer is exercised end to end: signature construction, blake2b
hashing, dispatch-key derivation, process cache, replay lookup, native fallback
and miss recording all run on real hardware.

Not yet done: record mode (HI10, needs a SQLite3 dev package), the tuning
engine (HI12), the fused collection points (HI13, now ahead of HI12), the
resource blacklist (HI09b), and the test suite (HI16).

gfx1201 and every multi-GPU topology are unverified — they need the server.
