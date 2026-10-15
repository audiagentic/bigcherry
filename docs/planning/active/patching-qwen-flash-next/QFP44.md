---
id: QFP44
order: 43
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-08T00:00:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Follow-up MMQ tile geometry and compact MoE block-map qualification

## 2026-10-15 authoritative b11552 audit: upper-bound compaction, completed Stream-K, no second map (BCOP228)

**Disposition:** QFP44 remains pending only for a **receipt-first compact MoE grid / tile-I-J/LDS headroom decision**. The 2026-10-08 wording below is historical where superseded. On `main` at `08441aa4`, `1237_rd30_moe_mmq_compact_grid` (gfx1100) and dependent `1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2` (exact gfx1201/gfx1030) are both manifest-`validated` and included in `config/recipes.toml` validated enhancements. `1350_mmq_few_tile_streamk` is **also validated and present on main**, opt-in for ordinary Q8_0 under the Flash-Next profile; QFP37 has moved to `docs/planning/completed/patching-qwen-flash-next/QFP37.md`. The proposed `1351_iq4xs_expert_i32` package is **absent**; do not present it as implemented or validated. No new MMQ implementation is authorized by this audit.

### Source-grounded execution and lifetime

- `engines/llamacpp/patches/1237_rd30_moe_mmq_compact_grid/patch.py` injects `ggml_cuda_mmq_moe_compact_max_blocks`, `mmq_build_moe_block_map`, and the non-Stream-K branch of `launch_mul_mat_q` into `ggml/src/ggml-cuda/mmq.cuh`. The existing `mm_ids_helper` produces `expert_bounds` before J selection; no second expert-ID/routing producer is needed. The prep kernel computes `block_start[e+1]-block_start[e]=ceil((expert_bounds[e+1]-expert_bounds[e])/J)` and fills `block_expert[m]`. Stream order is prep -> MMQ on the same `cudaStream_t`; `ctx.pool(id)` owns the temporary map buffers until launch completion under the existing pool lifetime. The separate `hip-autotune-dispatch.cu::ggml_hip_mmq_workspace` edit includes the map bytes in the candidate workspace upper bound.
- **Correct the launch-count claim:** the host does **not** know `block_start[n_experts]` without a GPU readback. It launches `grid.y = B = ceil((ncols_dst + n_experts*(J-1))/J)`, an **upper bound**, not the real `R=sum_e ceil(count_e/J)`. The kernel exits when `blockIdx.y >= block_start[n_experts]`; therefore `B-R` padded blocks are launched but do not execute MMQ tile arithmetic. Do not describe 1237 as launching exactly R blocks or count padded blocks as missing work. The bounded `B<65535` and per-block shared-memory checks fall back to the existing rectangular grid; the source has no host readback, dynamic launch or new synchronization.
- **Host-only discriminator (not HIP):** 10,000 seeded expert-count/J cases plus six deterministic test cases passed. At `n_experts=256,J=128,ncols_dst=256`, one expert receiving 256 assignments yields `B=256,R=2` (254 padded), eight experts receiving 32 each yield `B=256,R=8` (248 padded), while 256 experts receiving one each yield `B=R=256`. An all-zero distribution has `B=254,R=0` in the host formula. This is an accounting/dispatch-ceiling result, **not** measured GPU time, evidence that zero-row graphs reach the kernel, or a reason to add a CPU synchronization to discover R. Map allocation upper bound is `4*(n_experts+1+B)` bytes (ignoring allocator rounding), about 2 KiB for the B=256 case; it is not a new VRAM bottleneck.
- `1265/patch.py` widens the same compact gate to **exact** gfx1201 and gfx1030 IDs, not all RDNA4/RDNA2. `1350/patch.py::launch_mul_mat_q` requires `ids_dst == nullptr && expert_bounds == nullptr`, Q8_0, fallback, `GGML_PREC_Q8`, physical underfilled tiles, bounded fixup and exact gfx1100/gfx1201. Thus it does not replace MoE compaction; no combined 1350+MoE Stream-K experiment is admitted. `1351` is a **separate unimplemented I=32 MoE idea**, not a reason to change 1237's map or add a second selector.

### Measured history, not new gains

- `1237/evidence/validation.json`: final four-session b11126 gfx1100 contract record is eligible/validated; pp512 lane geometric effect **+7.32%** (CI95 within-run ~[+7.16%,+7.52%]), with a separate near-neutral tg128 control and byte-identical 256-expert test-backend-ops evidence. Earlier README manual pp512 +0.7–0.9% is a **different older composition/model measurement**; do not pool or replace the formal receipt.
- `1265/evidence/validation.json`: eligible b11126 gfx1201/gfx1030 records exist; outcomes are model/architecture dependent, including ~+1% MoE pp512 lanes, and some invalid/incomplete historical records. gfx1030's 16 GiB card used a smaller 32-expert model for speed while 256-expert correctness was synthetic. Do not generalize gfx1100's +7.32% to either device.
- `1350/SUMMARY.md`: separate b11474 Flash-Next ordinary Q8_0 prefill +1.8%/+1.2%/+1.9% at 8K/24K/98K; default off outside its qualified profile. Stream-K changes F32 reduction order. None of these are current-pin b11552 measurements or gains from QFP44's residual work.

### New upstream/fork gates (source/diff inspected 2026-10-15)

1. llama.cpp [#30553](https://github.com/ggml-org/llama.cpp/pull/30553) (open, 2026-10-14) caps **Stream-K** block count at actual available K work, with a separate whole-iteration bound and native fixup. It edits `mmq.cuh::launch_mul_mat_q`, **not** the 1237 compact MoE prefix/map. Its RTX 5090 decode numbers do not transfer to gfx1100/gfx1201; if adopted upstream, test composition with 1350 via the completed QFP37/1350 owner, not another QFP44 Stream-K patch.
2. llama.cpp [#28195](https://github.com/ggml-org/llama.cpp/pull/28195) merged 2026-10-14; its I=128 -> 64 table changes are **gfx115x RDNA3.5**, not gfx1100/gfx1201. The documented `ceil(ceil(M/I)*ceil(N/J)/resident_slots)` wave-round model is a useful *discriminator* for a hypothetical I=32 MoE case, not a transferable speedup or reason to edit production CASE rows.
3. llama.cpp [#30477](https://github.com/ggml-org/llama.cpp/pull/30477) remains an **unmerged PoC** for duplicate top-k expert IDs, changing `mmid.cu` and the maximum expert width. A compact-map test that starts from already-collapsed `expert_bounds` cannot prove correctness for duplicate IDs. Require an independent ordered-ID/weight/inverse-map oracle upstream of `mm_ids_helper`; if the stock path is wrong, assign that to the base expert-ID owner, **not** a second QFP44 map. No duplicate-ID route is declared qualified.
4. [AMD-Ecosystem/llama.cpp #63](https://github.com/AMD-Ecosystem/llama.cpp/pull/63) itself documents the same host upper-bound grid and GPU tail guard; [vLLM `moe_align_block_size`](https://docs.vllm.ai/en/v0.16.0/api/vllm/model_executor/layers/fused_moe/moe_align_block_size/) similarly returns a device-side padded count while allocating by a host bound. Reuse this principle; a dynamic exact-grid proposal would need a demonstrated net benefit after readback/graph overhead. No NVIDIA/MI-series speedup is evidence for BigCherry.

### Only remaining implementation/qualification gate

1. **Receipt census, no new hardware job yet:** reuse existing `1237/1265` contracts, `tools/bigcherry/tuning/catalog.py::enumerate_mmq/read_mmq_config_table`, `ggml_cuda_mmq_get_config`, `launch_mul_mat_q` and available rocprof/patch-hit traces. Per physical gfx1100/gfx1201/gfx1030 rank and actual `MUL_MAT_ID` route record `type, I, J, fallback, nrows_x, ncols_dst, ncols_max, expert_bounds distribution, B, R, nty, legacy ntx*n_experts, map-kernel us, MMQ us, pool bytes, LDS, VGPR, waves/CU, graph capture, stream and exact patch digest`. Treat `BIGCHERRY_PATCH_HIT` as route entry only; it does **not** report R or prove a saved kernel launch.
2. **Host-only terminal discriminator:** test single-hot, 8-hot, Zipf, uniform, all-zero, empty/tail and 1..4096 token counts against a CPU expert-count oracle. Assert `0<=R<=B<65535` on admitted cases, monotonic `block_start`, exactly one expert per real block, no read of uninitialized `block_expert[m>=R]`, correct per-rank `-1`/inactive IDs and no missing/duplicated work. For duplicate expert IDs, compare the full pre-`mm_ids_helper` oracle first; if stock differs, terminal `BASE_ROUTING_UNQUALIFIED`.
3. **Stop without patch** if the recorded MoE MMQ critical-path share is <3% E2E or the optimistic removable launch/padding share is <3% E2E. The upper-bound block count alone cannot establish that ceiling: padded blocks already exit early and a new dynamic launch may add synchronization. Do not create another grid, scheduler, map, buffer pool, selector, telemetry or CASE table. The only conditional I=32 experiment belongs to a re-opened QFP37 owner and requires a positive physical `ceil(ceil(M/I)*ceil(N/J)/resident_slots)` wave-round prediction, no new map, and available hardware lane.
4. **If an eligible residual appears:** single patch-owner opt-in, same-pin same-build off/on with 1237+1265 fixed and 1350 controlled off for MoE; gfx1100 then gfx1201 independently, gfx1030 only with a model-fit matching route. Include 8K/24K/98K, pp512/pp2048, tg128 negative, multi-request/ubatch, long-context, graph replay, Q4_K/Q8_0/IQ4_XS as actually dispatched, ordered expert IDs/weights, full-vocab logits/KLD/greedy, MTP acceptance, exact work and memory accounting. Require >=4 independent sessions and >=10 ABBA pairs/session, CI95-low >=3% **E2E** prefill and <=1% unaffected regression; reject if a single expert/tile disappears, routing changes, fallback bypasses validated gates or physical bandwidth is implausible. No experiment queued by this audit.

**Ownership:** QFP44 owns the residual MoE map/geometry *decision and receipts*; 1237/1265 own the validated map and exact architecture gates; completed QFP37/1350 owns ordinary Stream-K; original PRBE22/23/24 are consolidated historical proposals; 1281/1283 own expert-ID semantics; `0300` and existing MMQ catalog own J/config selection. Active MEN10, RHP/Radiance, RWQ and PGC15 paths are protected and unchanged.

**Validation this audit:** 12/12 source/manifest assertions and 6/6 host-only tests (10,000 randomized distributions); no repository suite, HIP compile, patch composition, GPU benchmark or new measured gain.


## Historical 2026-10-08 description (superseded where conflicting)

Consolidates PRBE22 (RDNA MMQ tile width/LDS), PRBE23 (MoE-aware tile selection) and PRBE24 (GPU compact block-map) beside QFP37. This is **residual work only**, not a new implementation of already validated functionality. Source target: composed llama.cpp b11474, gfx1100/gfx1201 HIP, physical post-Meta tensor shapes.

## Already covered (do not duplicate)

- `1237_rd30_moe_mmq_compact_grid` is `validated` for gfx1100; `1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2` is `validated` for gfx1201/gfx1030. Both are members of `[patch-set.validated-enhancements]`. The `mmq_build_moe_block_map` kernel builds `block_start[]/block_expert[]` from `expert_bounds` and compacts non-Stream-K expert/J work; there is no need to reimplement the map.
- `1345_moe_ids_multiwarp` is `validated`, in production, and its measured Flash-Next prefill improvement was +1.6-2.1%; this optimizes expert-ID handling, **not** the output-row I geometry.
- QFP37's `1350_mmq_few_tile_streamk` has owner-reported +1.2-1.9% prefill on 8K/24K/98K (PR #22 hardware comment); it is not present on `main` at this triage, so do not call it validated here. Its initial eligibility is ordinary non-MoE few-tile Q8_0, not a replacement for compact MoE grids.

## Historical remaining steps (superseded by 2026-10-15 gates)

1. Re-census `ggml/src/ggml-cuda/mmq.cuh::launch_mul_mat_q`/ `mul_mat_q_switch_J` on composed b11474 for `(gfx, quant type, physical M/N/K, nrows_x, J, I, experts, tiles, CU count)` and kernel time, with `1237/1265/1345` on and `1350` controlled separately. Include Flash-Next UD-IQ4_XS and 27B Q8_0.
2. **Tile width/LDS (PRBE22):** parse current `mmq-config-rdna{3,4}.cuh` CASE rows via `tools/bigcherry/tuning/catalog.py::enumerate_mmq/read_mmq_config_table`. CASE keys are `(type,J,fallback)`; replace the selected candidate row, never append a duplicate unreachable key. Sweep J and nthreads/occupancy/LDS for the measured shapes. Record rocprofv3 VGPR, LDS, waves/CU and kernel us; no global table swap.
3. **MoE output-row I (PRBE23):** only if compact-map profile still shows under-utilization, qualify QFP37's I=32 IQ4_XS MoE variant against native I=64/128 for physical rows, keeping the validated 1237/1265 map and 1281/1283 expert-ID semantics unchanged. Do not combine Stream-K for MoE until real expert-bound tile-count accounting exists.
4. **Block-map regression (PRBE24):** instrument the existing `mmq_build_moe_block_map` launch only in a validation build; dump `block_start[0..n_experts]` and `block_expert[0..actual_blocks)` after an opt-in sync. Compare to CPU enumeration for uniform/single-hot/Zipf/skew, empty/tiny/256-expert routings, token counts 1..4096. Independently record map kernel us, syncs (production zero additional), scratch bytes and overflow/fallback marker.
5. Preserve F32 equivalence for Stream-K (changed reduction order), byte-identical I-geometry outputs when K order is unchanged, independent test-backend-ops parity, greedy identity, and native fallback for ineligible shapes. Measure end-to-end via `tools/lab/flash-next/queue-env-ab.sh` in fully separated ABBA: 8K/24K/98K prefill, kernel time, peak VRAM, decode control and exact patch-hit shape. Promote narrowly only on reproducible per-shape gain with <=1% unaffected regressions.

## Scope and handoff

QFP37 owns Stream-K (1350) and I=32 (1351) implementation/qualification. QFP44 owns residual tile/LDS catalog sweeps and formal compact-map regression evidence; no duplicate patch packages. If 1237/1265 + 1350/1351 leave no material under-occupancy, close this item as a measured null.

## Change Log

- 2026-10-08 (triage): Created by folding PRBE22/PRBE23/PRBE24; referenced validated 1237/1265/1345 and open PR #22/1350 separately.
