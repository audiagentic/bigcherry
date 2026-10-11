# 1260_nro09_meta_view_headroom

**Status:** untested
**Plan item:** PNRO09

> Historical b10901 capacity (C) and real-graph (D) observations, not current b11474 qualification.

## Current-pin audit / superseding status (2026-10-09 UTC)

**Untested, producer non-startable, no capacity promotion.** The b10901 44/222 and 89/445 fixture results below are genuine historical observations but cannot be extrapolated to b11474: upstream [#23671](https://github.com/ggml-org/llama.cpp/pull/23671) replaced `ggml_get_mem_size(ctx)` with `n_tensors * ggml_tensor_overhead()` in `ggml_backend_meta_buffer_type_alloc_buffer_n`. New per-compute-container first-order capacity is `headroom * n_static` view objects, not `headroom * S / GGML_TENSOR_SIZE`. The 4B MTP workload fit at headroom 16, so the necessity of 80 is unproved. The 98.3 vs 76.5 t/s historical comparison was confounded; it proves neither improvement nor absence of regression.

`validation/producer.py` omits required `targets` for `build_materialized_pair`, assumes unmaterialized trees and a missing CMake target, classifies unrelated child failures as boundaries and copies old hardware evidence into a new artifact. The contract also requires controls that are not produced. `test_plan_producers.py::_KNOWN_NOT_STARTABLE` correctly blocks it. Replace with a current-pin CPU+Meta boundary and two-reset fixture, a fail-closed process oracle, actual per-buffer n_static/high-water accounting and a reproducible stock-fail/subject-pass real graph. Otherwise close PNRO09 and retire 1260 unpromoted. Authoritative details: PNRO09; disposition: BCOP94. New upstream [#30217](https://github.com/ggml-org/llama.cpp/pull/30217) is a separate host-view correctness baseline.

## What it does

Raises the multiplier `compute_headroom` from 16 to 80 in Meta's
`alloc_buffer_n` compute-container metadata allocation. Under b11474,
the reservation is `H * n_static * ggml_tensor_overhead()` per rotating
container and simple backend. The previous global "16-view limit" claim
was incorrect; the historical 72-view estimate is not a demonstrated
stock capacity failure.

## Why

A larger multiplier could prevent metadata-pool exhaustion if per-buffer
external-view high water exceeds `16 * n_static`. No such supported
stock-fail/subject-pass workload has been shown on b11474. This is a
host metadata-capacity hypothesis, not a performance claim.

## Upstream

Local (origin `local`); PNRO09. Anchored on the `constexpr size_t
compute_headroom = 16;` declaration in `ggml-backend-meta.cpp`.

## Historical b10901 hardware-free capacity boundary (C) — not b11474

The binding constraint is the ggml object pool (`ggml_new_object` enforces
`ctx->mem_size`); the headroom bounds tensor-object **metadata**, not model
data. A C++ fixture (`validation/meta_boundary_test.cpp`) drives the REAL Meta
backend (CPU simple device -> meta device -> meta backend) with one static leaf
(size S) plus N between-eval view tensors of the leaf, registered into
`stc_compute`. Each view consumes one fixed `GGML_TENSOR_SIZE` metadata object,
so `N_max = compute_headroom * S / GGML_TENSOR_SIZE` (proportional to headroom).

Observed `N_max` (unpatched headroom=16 vs patched headroom=80):

| S (bytes) | N_max (16) | N_max (80) | ratio |
|-----------|-----------|-----------|-------|
| 1024      | 44        | 222       | 5.045 |
| 2048      | 89        | 445       | 5.0   |

The ratio (~5 = 80/16) is confirmed at two independent S values, with a bounded
linear metadata cost (`stc_compute` mem_size = headroom * S; 80x vs 16x the
static mem_size) and a safe abort on capacity exceedance (`ggml_new_object:
not enough space ... needed 164128, available 163840`). See
`evidence/boundary.json`. This is a capacity (correctness-scoped) result, not a
performance claim.

## Historical b10901 real-graph (D) hardware lane — not b11474

Built the patched (headroom=80) and unpatched (headroom=16) variants on Brutus
(gfx1100 7900 XTX devices 0/1) and ran the real recurrent+MTP graph —
tierA-qwen4b-q6k MTP (`--spec-type draft-mtp --spec-draft-n-max 4`) — on both.

| variant | result | prompt t/s | gen t/s | estimated 72 views |
|---------|--------|-----------|---------|----------|
| patched (headroom=80) | MTP decode to completion, NO abort | 21.9 | 98.3 | FIT |
| unpatched (headroom=16) | MTP decode to completion, NO abort | 18.6 | 76.5 | ALSO FIT |

Both variants completed the historical 4B MTP run, so the workload did
not establish the need for 80. The 72-view count is a model-derived
estimate, not a measured per-buffer high-water mark. The throughput gap
was confounded and cannot prove improvement or non-regression. The
historical capacity formula depended on user-context size S; it does not
describe the b11474 allocator. See `evidence/hardware_d.json`.
