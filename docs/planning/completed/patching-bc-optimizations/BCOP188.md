---
id: BCOP188
order: 188
plan: patching-bc-optimizations
state: completed
created-at: '2026-10-13T15:18:00Z'
created-by: agent
priority: P2
work: S
---

# PKC05: qualify Q1_0 native HIP unpack without duplicating kernels

## Description

**Discovery/change:** Pinned llama.cpp b11552 has a ten-`__byte_perm` unpack chain per 16-bit Q1_0 word at both `vecdotq.cuh::vec_dot_q1_0_q8_1` (MMVQ) and `mmq-load-tiles.cuh::ggml_cuda_mmq_load_tiles_q1_0` (MMQ). Open upstream #29927 (head `b6c4ff7`) replaces each with four HIP-native `__builtin_amdgcn_perm` calls. Host-only exhaustive signed-16-bit mapping (65,536 patterns), 4,096 signed dot products, six boundary patterns and 12 source assertions passed; **no GPU compile/benchmark or speedup**.

**Owner/consolidation:** PKC05 remains the sole quant-kernel owner. Existing HIP autotune owns dispatch; QFP44 owns tile geometry; PRBE111/1273 owns unrelated IQ kernels. #28398 is a competing decode-only unpack, **not** a second patch to compose. The six registered BigCherry contract models contain no Q1_0, so production throughput cannot be inferred. Recent Radiance GPU-mailbox/expert-helper, patch-prefix and other 12-hour active capabilities were excluded and untouched. No new registry, scheduler, kernel family, allocator, telemetry or hardware queue.

**Unresolved action:** Check exact #29927 applicability/patch composition against b11552 and 1273, then compile gfx1100/gfx1201 and run existing Q1_0 backend-op/ISA correctness controls. Require a real hash-bound Q1_0 model and executed MMVQ/MMQ route before any model A/B; if absent, `AWAIT_Q1_MODEL`. Reject if mapping, backend-op, compiler resources or safety fail, or if measured exposed-kernel time cannot theoretically yield 3% E2E. Otherwise require >=4 independent-session paired AMD campaigns with >=3% E2E gain, positive CI95 lower bound, <=1% controls and full numerical/graph safety. Merge upstream instead of maintaining a duplicate if #29927 lands.

**Evidence/dependencies:** llama.cpp b11552 (`23b0202a`); [#29927](https://github.com/ggml-org/llama.cpp/pull/29927), [#28398](https://github.com/ggml-org/llama.cpp/pull/28398); `config/recipes.toml`, `config/models.toml`; PKC05 authoritative details. Host fixture Python compilation passed. No repository pytest, patch application, HIP build, GPU run or new performance measurement.

## Steps


## Detailed Solution & Technical Design


## Code Samples & Guidance


## Files


## Validation



Source-static and host-only checks recorded in this item's disposition/evidence text; no hardware run was required for this closure.

## Effort & Risk


## Standards


## Acceptance Criteria



The disposition and ownership decisions described above are recorded; no further implementation is owed by this item.

## Notes


## Change Log

- 2026-10-23: state 'done' (invalid) corrected to completed
