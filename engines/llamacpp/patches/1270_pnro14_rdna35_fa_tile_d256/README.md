# 1270_pnro14_rdna35_fa_tile_d256

**Status: untested; no BigCherry gfx1151 hardware qualification.** PNRO14 is authoritative; BCOP74 records the 2026-10-09 route/architecture rebaseline.

The existing patch overrides only the D=256 / ncols=32 tile-FlashAttention row from `(256,256,32,256,3,64,128)` to `(256,256,32,256,4,64,64)`. It does **not** force the TILE backend. In b11474 `ggml_cuda_get_best_fattn_kernel` chooses AMD MMA before TILE for normal aligned, masked GQA prefill; `launch_fattn_tile_switch_ncols1` must also select 32 columns for this override to run. The once-only `BIGCHERRY_PATCH_HIT` marker records host configuration selection, not launch count or a measured gain.

The current `RDNA3_5`/host `GGML_CUDA_CC_IS_RDNA3_5` gates cover gfx1150/1151/1152/1153, **not just gfx1151**, although patch.toml lists only gfx1151 for validation. This mismatch blocks promotion; either narrow both selectors to an exact qualified ISA or qualify every admitted architecture. gfx1100/gfx1201 must remain native.

PNRO19's two anchor bugs were fixed in independent commit `f4228540b7` on 2026-09-27. The patch remains untested for performance. AMD downstream PR #41 reported gfx1151 pp1024 +0.3% with `-r 1` (noise), not BigCherry evidence. No hardware build/benchmark ran in the 2026-10-09 audit.

References: https://github.com/AMD-Ecosystem/llama.cpp/pull/41 ; https://github.com/ggml-org/llama.cpp/discussions/26377 ; https://github.com/ggml-org/llama.cpp/pull/26046 .
