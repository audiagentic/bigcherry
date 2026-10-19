---
id: RCS09
order: 9
plan: build-radiance-components
state: pending
created-at: '2026-10-19T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# No model constants in the helper: model shape from the container, model settings from an external profile

## Description

Owner direction 2026-10-19: do not tie the helper (or any BigCherry radiance component) to a specific model. Today moe_remote_shm.h hard-codes Flash-Next:
kHidden 2560, kTopK 10, kExperts 512, kMaxLayers 128, kMaxRanks 2 and the activation group, and the shared-memory header rejects any other shape on purpose
(fail closed). Different quants of the same model are already fine (rows per step do not depend on weight format); a different MoE model, expert count, top-k or
hidden size is not. Two kinds of fact are tangled in the code and must be separated:

1. MODEL SHAPE (a property of the container): hidden size, experts, top-k, layers, expert intermediate width, shared experts, activation group, weight formats.
   Read from the .rad container metadata at helper start and written into the shm Header as runtime values; engine and helper compare them at attach (fail closed
   on any mismatch, as now). Buffer capacities become bounds derived from the shape (or a compile-time maximum with an explicit, checked limit), not model-named constants.
2. MODEL SETTINGS (tuning, not shape): lane sizes, row limits, GEMV/grouped crossovers, ownership budget and heat file, helper count and devices, expert pool size.
   Externalised in the model's runtime profile (src/profile/*.ini and the recipes' runtime-profile tables), keyed by model, not baked into code or into a lab config.

Profiles hold only what differs from the defaults (owner rule 2026-10-05), and anything derivable is derived rather than configured (owner rule 2026-10-19): the defaults come from the
shape and the serving flags (--max-num-seqs, speculative depth), see RCS10; a profile key is justified only by what cannot be derived.

## Steps

1. Inventory every model-specific constant and assumption in libr3/helper and the rad0002/rad0004 hooks (kHidden, kTopK, kExperts, kMaxLayers, 2560-wide staging, 512-entry
   tables, E4M3/group 128 activations, W4 layouts) and classify each as shape, capacity bound or tuning.
2. Header v14: shape fields filled from the container; validate() compares them; buffers sized from the shape with a checked maximum; kernels take shape as arguments or
   template parameters selected at load (a small set of instantiations, not a per-model fork).
3. Move tuning knobs behind a profile section with a documented schema and offline validation (bigcherry check): unknown keys fail, ranges are checked.
4. Tests: the shm test and the replay harness run with a second synthetic shape (different hidden, experts, top-k) and must pass bitwise against the CPU reference; a model whose shape
   exceeds the checked maxima is refused with a clear message.
5. Document the model-onboarding checklist in docs/reference/engines/radiance (what to add for a new model: container, profile, shape check, tuning run).

## Validation

Shape read from the Flash-Next container equals today's constants (bitwise identical replay, unchanged acceptance numbers); a synthetic second shape passes the replay
and shm tests; no model name or model constant remains in the helper sources (a lint rule, in the spirit of the engine-name literal lint).

## Notes

Part of the component split (RCS04/RCS05): the moe-remote and backend components take their shape from the header, not from compile-time model constants. Related: RAD11 (other
expert quantisations), RCS07 (other architectures), RCS10 (tuning with the release build).

## Staged conversion plan (GPT review 2026-10-19)

Classify, do not mechanically replace: **protocol capacity bounds** (max rows, layers, routes, ranks) stay as checked maxima; **model shape** (hidden, top-k, experts, activation group, expert width, layers, strides, ownership-map strides, expert flattening) becomes runtime; **kernel tile geometry** (block dimensions, wave geometry, GEMM tiles, vector widths, bounded scan widths) stays compile-time. The literal `512` is three different things (expert count, thread count, launch-grid cap). Occurrence counts of the model constants plus `512`: moe_remote_gpu.h 219, r3_moe_worker.cpp 97, moe_remote_shm.h 39, moe_mailbox_ops.hip 23, moe_remote_hook.cpp 20, moe_remote_shm_test.cpp 4.

Architecture: runtime shape for orchestration and transport, a small set of shape-specialised GPU implementations chosen by dispatch (runtime dimensions with compile-time tiles, as CUTLASS and Composable Kernel do). Keep the Flash-Next 512/10 fast path; add generic paths beside it.

| PR | Scope | Gate |
|---|---|---|
| RCS09-01/02 | `LayoutShape`, checked `layout_for(shape)` beside the constexpr `layout()` | legacy shape reproduces v13 byte for byte; overflow and malformed-shape refusals |
| RCS09-03 | Header v14 (`intermediate`, `activation_group`, shape and capacity hashes), fail-closed attach, request validation | stale v13, wrong shape, wrong capacity, malformed offsets rejected |
| RCS09-04 | Runtime expert and layer indexing in helper, hook, CPU reference, ownership and heat | Flash-Next replay bitwise; synthetic 256-expert/2048-hidden tests |
| RCS09-05 | Runtime route counts, top-k, GPU routing; keep the 512/10 specialisation | differential routing and ownership tests; per-row-regime benchmark |
| RCS09-06 | Runtime hidden, activation group, strides, staging, quantisation, result reduction | BF16/FP8 operand comparison; canary/ASAN; bitwise Flash-Next replay |
| RCS09-07 | Runtime expert width, GEMV K, backend scratch and GEMM dispatch | widths 512 and 640; backend probe; prefill and decode benchmarks |
| RCS09-08 | Remove remaining model constants and wrappers; captures v3; docs | full regression matrix |

Hardest sites: the fixed 512-expert block scans and `__launch_bounds__` (moe_remote_gpu.h 189-244), `kGemvMaxK` shared-memory tile capacity (361-383), quantised-row chunk counts (125-127, 264-265), shared arrays sized by `kTopK` (537-600; moe_mailbox_ops.hip 365-366), fixed owner and layer-map capacities and `layer * kExperts` bit-index arithmetic (moe_remote_shm.h), route maps and histograms in the hook (342-415), the 640-wide reference arrays in r3_moe_worker.cpp (298-410), and heat and capture dimensions (605-737).

Traps: a smaller hidden does not reduce LDS use while static arrays stay sized for 2560; the expert-width 512 versus 640 assumption sits inside the GEMM kernel layout and output strides; the activation quantisation group (128) is not the weight tile `block_k`, and a hidden that is not whole groups must be refused, not truncated; one-thread-per-expert routing does not generalise (use a fixed tile with grid-stride); the owner bitmap stride change touches every CPU and GPU consumer; capture format is versioned separately from the shm protocol; accumulation order decides bitwise equality, so preserve it for the legacy specialisation.
