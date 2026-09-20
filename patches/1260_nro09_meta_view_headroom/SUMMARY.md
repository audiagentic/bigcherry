# 1260_nro09_meta_view_headroom

**Status:** untested
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
