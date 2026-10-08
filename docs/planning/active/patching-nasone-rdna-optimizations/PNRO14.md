---
id: PNRO14
order: 0
plan: patching-nasone-rdna-optimizations
state: completed
created-at: '2026-09-09T10:53:13.365873+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: P2
---

# gfx1151 D=256 tile FlashAttention configuration — retired without promotion

## Disposition (2026-10-08)

**Completed: do not promote or rework patch 1270 on the present BigCherry fleet.** This is a terminal *no-qualified-lane* decision, not a claim that the gfx1151 tile is slower or that hardware correctness passed. The gfx1151-only candidate is not selectable on gfx1100, gfx1201, or gfx1030; BigCherry has no gfx1151 hardware. Retain the existing untested package for provenance. Do not reserve another patch ID, create a second attention selector, or run unrelated fleet hardware as a proxy.

## Implementation and eligibility (pinned llama.cpp b11474, commit b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea)

- `ggml/src/ggml-cuda/fattn-tile.cuh::ggml_cuda_fattn_tile_get_config_amd_rdna`: baseline `(DKQ,DV,ncols,nthreads,occupancy,nbatch_fa,nbatch_K)=(256,256,32,256,3,64,128)`.
- `patches/1270_pnro14_rdna35_fa_tile_d256/patch.py` introduces a gfx1151-only `(256,256,32,256,4,64,64)` row, with runtime host `GGML_CUDA_CC_IS_RDNA3_5(cc)` and compile-time device `RDNA3_5` gates. Other shapes and architectures fall back to the generic table. Host/device config consistency is necessary because `flash_attn_tile` consumes compile-time `__launch_bounds__` and tile geometry, while host launch code chooses threads and batches.
- **Dispatch boundary:** `ggml/src/ggml-cuda/fattn.cu::ggml_cuda_get_best_fattn_kernel` selects MMA before TILE. For D=256, `amd_wmma_available(cc) && gqa_opt_applies && Q->ne[1]*gqa_ratio_eff>16` selects `BEST_FATTN_KERNEL_MMA_F16`. The 1270 tile row is then not executed. It may remain reachable on other eligible tile-routed shapes; do not claim universal dead code without a dispatch trace. `ggml_cuda_flash_attn_ext` dispatches TILE versus MMA, and `ggml_cuda_flash_attn_ext_tile_case` / `launch_fattn` are the tile launch path.
- **Trace limitation:** patch 1270's `BIGCHERRY_PATCH_HIT` marker is emitted by the host *configuration lookup*, not immediately after a successful tile launch. It proves a matching config lookup, not completed kernel execution, correctness, or speedup. Validation must separately prove actual TILE launch, `ncols=32`, the intended device, and absence of an MMA dispatch for the measured workload.
- `ggml/src/ggml-cuda/fattn-tile.cuh` at inspected upstream master retains the same generic D256/ncols32 row and host/device RDNA dispatch as b11474. No upstream adoption of this particular 1270 row was found.

## Superseded PNRO19 and patch mechanics

`PNRO19` described two real historical anchor failures (include span 3 vs 2 lines; device selector literal comments removed by the patcher's noise stripping). Commit `f4228540b774afee0efb2f8e31e760b0a3c65f67` on 2026-09-27 already corrected them with `max_span_lines=3` and `csource.strip_noise(_DEVICE_OLD, "c")`. The current `tools/tests/patch/test_1270_pnro14_rdna35_fa_tile_d256.py` contains apply/idempotence and missing-selector fail-closed tests. PNRO19 is completed as a stale bug report; do not re-author these anchors. No tests were executed on hardware or in a BigCherry checkout during this audit.

## Evidence and ownership

- **BigCherry measurement:** no first-party gfx1151 activation, correctness, or E2E A/B exists. Non-target gfx1100/gfx1201 measurements cannot qualify this arch-exclusive change.
- **External, different mechanism:** the gfx1151 Strix Halo fork `justinappler/llama.cpp-strix-halo` reports upstream's newer MMA path outperforming its own custom TILE patch by **11.2% pp at 16K context** (2026-09-25); this is not a direct 1270 comparison and cannot establish 1270's effect. It is a reason to prove the selected backend before tuning an unselected tile row.
- Upstream PR #22880 introduced RDNA3 MMA FA; PR #28102 subsequently tuned MMA and fixed head-256 selection. Neither is evidence that patch 1270 improved gfx1151.
- PNRO14 exclusively owns this gfx1151 tile-row candidate; upstream HIP FA retains dispatch and kernel ownership. PNRO19 owns only the resolved patch-anchor report. Do not overlap current MTP/Flash-Next, QFP, or general FA kernel tuning.

## Reopen gate (only with new independent evidence)

1. A gfx1151 device and reproducible b11474-equivalent binary become available. Trace `ggml_cuda_get_best_fattn_kernel` and `launch_fattn` to prove **TILE** execution with `(DKQ,DV,ncols)=(256,256,32)`; capture nthreads, occupancy, batch-K, mask/GQA eligibility, quant/KV type, register/LDS usage and full launch counts. If target TILE time is <5% of production E2E wall time or MMA already wins, terminate without patch.
2. Compare exact baseline generic row versus 1270 at identical model/quant/context/batch/ubatch/FA/KV/clocks. Include D256 target prefill at shallow and >=16K context, decode, a non-target D128 control, and stable/repeated same-process and multi-ubatch requests. Verify full backend-reference logits/greedy identity, no NaNs, graph replay/allocation stability, identical attention work and no skipped kernels.
3. Use >=4 independent sessions and >=10 interleaved paired rounds per session. Promote only if **CI95 lower bound >=3% E2E** on the target, <=1% control regression, correctness and actual tile-launch proof. Otherwise reject 1270. Retain native fallback and do not introduce a runtime tuning table.

## Sources

- https://github.com/ggml-org/llama.cpp/blob/b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea/ggml/src/ggml-cuda/fattn.cu
- https://github.com/ggml-org/llama.cpp/blob/b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea/ggml/src/ggml-cuda/fattn-tile.cuh
- https://github.com/ggml-org/llama.cpp/pull/22880
- https://github.com/ggml-org/llama.cpp/pull/28102
- https://github.com/justinappler/llama.cpp-strix-halo/blob/master/strix-halo/fa-mma-d256-26419.md

## Validation / history

2026-10-08: 10 read-only source/patch/test assertions and two host-side packed-config arithmetic checks passed; no build, patch application, kernel, hardware or benchmark ran. Last independent PNRO14/1270 work: patch fix 2026-09-27; no independent work, submission or queued experiment in the preceding 12 hours. The 2026-09-24 proposal and 2026-09-26 patch are superseded by this disposition, not deleted.
