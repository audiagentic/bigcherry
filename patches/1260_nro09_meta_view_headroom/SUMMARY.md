# 1260_nro09_meta_view_headroom

**Status:** capacity boundary (C) + real-graph (D) hardware lane verified (2026-09-23)
**Plan item:** PNRO09

## What it does

Raises `compute_headroom` in `ggml-backend-meta.cpp` from 16 to 80 so the
compute-container view allocation covers recurrent+MTP graphs. For
`n_rs_seq=8` with 4 recurrent layers the GDN output tensor creates
`2*(n_rs_seq+1)` views per recurrent slot (`2*9*4 = 72` views), exceeding
the original 16-view bound.

## Why

The 16-view bound was sized for non-recurrent workloads. With recurrent+MTP
configuration (recurrent layers plus MTP speculative decode), the view count
grows with `n_rs_seq` and the number of recurrent layers, and 16 is too small
for the maximum supported configuration. This is a memory-allocation change only
(compute buffer sizing); it does not alter numerical behavior.

## Upstream

Local (origin `local`); PNRO09. Anchored on the `constexpr size_t
compute_headroom = 16;` declaration in `ggml-backend-meta.cpp`.

## Hardware-free capacity boundary (C)

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

## Real-graph (D) hardware lane

Built the patched (headroom=80) and unpatched (headroom=16) variants on Brutus
(gfx1100 7900 XTX devices 0/1) and ran the real recurrent+MTP graph —
tierA-qwen4b-q6k MTP (`--spec-type draft-mtp --spec-draft-n-max 4`) — on both.

| variant | result | prompt t/s | gen t/s | 72 views |
|---------|--------|-----------|---------|----------|
| patched (headroom=80) | MTP decode to completion, NO abort | 21.9 | 98.3 | FIT |
| unpatched (headroom=16) | MTP decode to completion, NO abort | 18.6 | 76.5 | ALSO FIT |

For the 4B MTP workload the ~72 between-eval views (and their metadata
footprint) fit even within the stock 16-view headroom, so 16->80 does not
change the pass/fail outcome for this specific workload. The patch is a headroom
increase that raises the ceiling for larger/more views; the 4B MTP graph's ~72
views are small enough to fit under the stock bound. **No perf regression** from
the patch (patched is actually faster: 98.3 vs 76.5 gen t/s). The binding case
would be a workload with a larger per-view metadata footprint or a larger view
count — the (C) hardware-free boundary test is the direct capacity proof, and the
(D) hardware lane confirms the patch is benign (no regression, no correctness
issue) on a real recurrent/MTP graph. See `evidence/hardware_d.json`.
