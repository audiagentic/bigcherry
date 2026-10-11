# 1260 (PNRO09): Meta compute-container view headroom

## Current-pin qualification correction (2026-10-09 UTC)

**Blocked / untested.** Upstream llama.cpp [#23671](https://github.com/ggml-org/llama.cpp/pull/23671) changed the Meta allocator on 2026-10-02. At pinned b11474, `ggml_backend_meta_buffer_type_alloc_buffer_n` reserves `compute_headroom * n_tensors * ggml_tensor_overhead()` **per rotating compute container**, not `compute_headroom * ggml_get_mem_size(user_ctx)`. The 2026-09-23 boundary values (44/222 at S=1024, 89/445 at S=2048) came from b10901 and are historical only; they are **not current-pin qualification**. A one-static-tensor fixture now predicts roughly 16/80 views regardless of S; measure real per-buffer static count and high water. The 4B MTP graph completed on **both** variants, so it did not prove a capacity deficit.

The current producer is explicitly non-startable (`tools/tests/patch/test_plan_producers.py`): its `build_materialized_pair` call omits required `targets`, assumes source trees and a nonregistered `meta_boundary_test` CMake target, treats arbitrary process failures as capacity boundaries, and re-emits historical hardware evidence without running the required decode/state-restore control. Fix the host fixture/provenance and contract first; no promotion, benchmark queue or performance claim. See authoritative PNRO09 and BCOP94. Upstream [#30217](https://github.com/ggml-org/llama.cpp/pull/30217) changes Meta host-view handling **after** the pin and requires independent rebase validation.

## Scope

Raises `compute_headroom` in `ggml/src/ggml-backend-meta.cpp` from 16 to 80
so the Meta backend's compute-container object pool can hold the
between-eval views that recurrent + MTP graphs create
(`2*(n_rs_seq+1)` views per recurrent slot). Memory sizing only; no
numerical change.

## Validation

Contract `PNRO09-META-VIEW-HEADROOM` (correctness/capacity), producer
`nro09`:

- capacity (C): `validation/meta_boundary_test.cpp` probes the largest view
  count `N_max` that fits for two leaf sizes on both variants; the patched
  ratio must be ~80/16 = 5;
- real graph (D): tierA-qwen4b-q6k MTP decode on both variants completes
  without a Meta-backend object-pool abort.

The recorded (D) throughput difference (98.3 vs 76.5 gen t/s) is NOT a
performance claim: a headroom change cannot explain a 28% decode gap, so
that comparison was confounded and must not be cited.

## Status

untested.
