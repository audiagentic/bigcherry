---
id: PNRO14
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:53:13.365873+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: P2
---

# RDNA3.5 D=256 tile FlashAttention: route- and architecture-gated disposition

**2026-10-09 audit (authoritative):** Patch 1270 exists and its two PNRO19 anchor bugs were fixed in f4228540b7 (2026-09-27). Keep it untested and out of production. Current fleet gfx1100/gfx1201/gfx1030 cannot qualify a gfx1151 tuning claim. No BigCherry E2E gain is measured. Historical 2026-09 notes below describe earlier states, not present work.

## Description

The only candidate is `patches/1270_pnro14_rdna35_fa_tile_d256`, derived from AMD-Ecosystem/llama.cpp PR #41 (merged downstream 2026-07-10). It replaces the native RDNA `(DKQ,DV,ncols,nthreads,occupancy,nbatch_fa,nbatch_K)=(256,256,32,256,3,64,128)` with `(256,256,32,256,4,64,64)` **only if the tile kernel actually launches with 32 columns**. Pinned b11474 and upstream master have the same `fattn-tile.cuh` blob (`b5ea915e`), and neither contains this RDNA3.5-specific override. The package's claim of *gfx1151-only* is imprecise: `GGML_CUDA_CC_IS_RDNA3_5` and `RDNA3_5` cover gfx1150/1151/1152/1153; only gfx1151 is listed for validation. Do not promote with that mismatch.

**Route-level discriminator:** `ggml/src/ggml-cuda/fattn.cu::ggml_cuda_get_best_fattn_kernel` selects AMD WMMA/MMA before TILE when `gqa_opt_applies` and `Q->ne[1] * gqa_ratio_eff > 16`. In `fattn-tile.cuh::launch_fattn_tile_switch_ncols1`, a 32-column tile requires `Q->ne[1] > 16/ncols2`. For ordinary aligned, masked GQA with ratios 2/4/8/16/32, those conditions overlap: cases large enough for tile32 are routed to MMA instead. Therefore the candidate may have **zero activation** on the target GQA prefill lane. MHA (ratio 1), missing/ineligible masks, or other exceptional shapes can reach tile32. This is source-level inference, not hardware timing.

AMD PR #41 measured pp128 +2.7%, pp1024 +0.3%, tg128 -0.1% on gfx1151 Qwen3.6-35B-A3B Q4_K_M with only one repetition (`-r 1`); not statistically qualified and not BigCherry evidence. Upstream #26046 removed rocWMMA FlashAttention, so AMD PR #40's separate rocWMMA-on mechanism is obsolete for this pin. External AITER and vLLM ROCm mechanisms are separate engines, not drop-in tile configurations.

## Steps

1. **Stop duplicate patch work:** keep package 1270 untested; PNRO19 is terminal because commit `f4228540b7` already fixed `max_span_lines=3` and comment-stripped device-selector anchors. Do not create another patch or run a new hardware lane on the current fleet.
2. **Cheap static gate (already executed):** compare b11474/master source, patch anchors and selector precedence; enumerate masked-GQA/MHA cases. If the candidate cannot reach tile32 on a model, close that model lane without benchmarking.
3. **Future gfx1151-only qualification, if such hardware becomes available:** first choose between narrowing both host/device gates to exact gfx1151 or qualifying all gfx115x members. The current broad gate plus gfx1151-only validation is insufficient. Confirm patched and stock binaries share the same pin/build flags, model and backend; `GGML_HIP_ROCWMMA_FATTN` is obsolete on b11474.
4. Capture real `BEST_FATTN_KERNEL_TILE` selection, `cols_per_block=32`, `DKQ=DV=256`, launch counts, per-kernel GPU time, VGPR/LDS/occupancy, and fraction of E2E prefill. Existing once-per-process `BIGCHERRY_PATCH_HIT` is **not** a per-launch counter or speed proof.
5. Only if tile32 contributes >=5% of E2E prefill, compare stock vs candidate using a positive MHA D256 case and masked GQA negative controls; 4 sessions x 10 paired ABBA rounds, pp128/512/1024/4096, KV 8K/80K, TG128/512. Reject if no CI95-low >=3% E2E prefill gain, any correctness failure, or >1% TG/control regression. Otherwise retain the untested/deferred disposition.

## Detailed Solution & Technical Design

**Existing code, no new owner:** `ggml_cuda_get_best_fattn_kernel` chooses VEC/MMA/TILE; `launch_fattn_tile_switch_ncols2` chooses GQA packing, `launch_fattn_tile_switch_ncols1` chooses 2/4/8/16/32/64 columns; `ggml_cuda_fattn_tile_get_config` supplies launch bounds and `nbatch_fa` on host and device. The patch's override changes occupancy and K-column tile only, not algorithm, KV ownership, graph topology, transfers or scheduler. Do not introduce a second attention dispatcher/telemetry system.

**Future selector pseudocode:** `if exact_qualified_arch && kernel==TILE && DKQ==DV==256 && cols_per_block==32: use candidate_config; else use native_config`. Current patch implements a broader RDNA3.5 architecture gate, not this proposed narrowed admission. Host/device must agree on the same config word; `GGML_CUDA_FATTN_TILE_CONFIG_CASE` packs threads/occupancy/nbatch_fa/nbatch_K in one uint32. Preserve native fallback for every other shape/arch. No P2P, RCCL or auxiliary-gfx1030 interaction is involved.

**Safety:** run FLASH_ATTN_EXT reference comparisons (greedy/logits or KLD as applicable), masked/unmasked, multi-ubatch, long context and repeated same-process requests. Confirm actual kernel counts/work and graph replay remain unchanged; reject apparent gains from skipped work. Include architecture negative controls gfx1100/gfx1201 and a gfx1030 non-selection check. Use existing profiling/lab primitives, not a new global dispatch/config table.

## Code Samples & Guidance

Current b11474: `ggml/src/ggml-cuda/fattn.cu::ggml_cuda_get_best_fattn_kernel`; `ggml/src/ggml-cuda/fattn-tile.cuh::{ggml_cuda_fattn_tile_get_config_amd_rdna,ggml_cuda_fattn_tile_get_config,launch_fattn_tile_switch_ncols1,launch_fattn_tile_switch_ncols2}`; `ggml/src/ggml-cuda/common.cuh::amd_wmma_available`; `ggml/src/ggml-cuda/vendors/hip.h::RDNA3_5`. Existing `patches/1270_pnro14_rdna35_fa_tile_d256/patch.py` contains four edits, including once-only host logging; do not recreate the three-edit proposal from September.

## Files

Authoritative: this PNRO14 plan. Existing patch package 1270 README/SUMMARY/patch.py/patch.toml and `tools/tests/patch/test_1270_pnro14_rdna35_fa_tile_d256.py`. PNRO19 is a resolved duplicate mechanics issue. BCOP74 is a thin disposition ledger. No new implementation files or benchmark definitions in this audit.

## Validation

**Executed 2026-10-09 (host-only source checks):** 11/11 assertions passed against exact b11474 `fattn-tile.cuh` and patch 1270; 132 selector-model cases enumerated (masked aligned GQA ratios 2..32: 0 tile32 cases; MHA Q17 and unmasked Q17 can reach tile32). These are static source/route fixtures, **not** compiled patch mechanics, GPU profiling or hardware qualification. Historical PNRO19 failures were fixed by independent commit `f4228540b7`; the actual BigCherry test suite was not executed in this run. Future offline commands: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1270_pnro14_rdna35_fa_tile_d256`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1270_pnro14_rdna35_fa_tile_d256 --source bigcherry-tuning`; `python -m pytest tools/tests/patch/test_1270_pnro14_rdna35_fa_tile_d256.py`. Hardware is unavailable in the fleet.

## Effort & Risk

No implementation justified now. Main risk is measuring the wrong attention path or enabling an unqualified gfx115x ISA. Once-per-process activation markers cannot substitute for per-launch attribution. No evidence establishes improvement on gfx1100/gfx1201.

## Standards

Fail-closed architecture admission; exact b11474 baseline; route proof before benchmarks; do not extrapolate single-run gfx1151 or AITER/vLLM measurements to BigCherry; no interference with current MTP/QFP/Meta lanes.

## Acceptance Criteria

Remain deferred unless an actual qualified gfx1151 (or separately qualified gfx115x) environment proves tile32 activation, exact host/device config parity, no non-target selection, full correctness, CI95-low >=3% E2E prefill benefit and <=1% TG/control regression. Close if tile32 <5% of E2E prefill, no activation, or hardware cannot be qualified.

## Notes

2026-10-09 authoritative rebaseline: the historical TODO/12xx instructions above have been replaced; 1270 already exists, PNRO19 was fixed in f4228540b7, and the unqualified architecture-wide gate is a release blocker. External: https://github.com/AMD-Ecosystem/llama.cpp/pull/41 ; https://github.com/ggml-org/llama.cpp/discussions/26377 ; https://github.com/ggml-org/llama.cpp/pull/26046 ; https://github.com/vllm-project/vllm/issues/54438 ; https://github.com/ROCm/aiter .

Supersedes: NRO15
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro15

2026-09-24 relevance at b11126: UPSTREAM-ABSORBED. Verified `ggml/src/ggml-cuda/fattn-tile.cuh`: `ggml_cuda_fattn_tile_get_config_amd_rdna()` already has a D=256/ncols=32 row (256,256,32,256,3,64,128), selected for any GGML_CUDA_CC_IS_RDNA(cc) architecture including RDNA3.5 (gfx1151) and this project's own gfx1100/gfx1201 (both RDNA umbrella members per common.cuh IS_RDNA3/IS_RDNA4). The item's literal narrow ask (a separate RDNA3.5-only nwarps=4 row with gfx1100/gfx1201 excluded) doesn't exist, but the functional capability it exists to provide is already shipped and already active on this project's hardware. No gfx1151 hardware exists in this project's fleet to produce the tuning evidence the item's acceptance criteria require anyway. GPT design consultation was not used -- disposition was resolved directly from source, no design question remained.

2026-09-24: reopened from superseded -- GPT review req_14f299d271894983 showed the RDNA3.5-specific row is NOT upstream (b11126 fattn-tile.cuh:61 generic RDNA row 256,256,32,256,3,64,128 in ggml_cuda_fattn_tile_get_config_amd_rdna; selectors at :317 host and the #ifdef RDNA device branch; RDNA3_5 macro vendors/hip.h:224, GGML_CUDA_CC_IS_RDNA3_5 common.cuh:91). Anchors verified against the b11126 mirror.

## Change Log

- 2026-09-09T10:53:13.365873+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:43.442305+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.111896+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.772814+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:46:10.318015+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024630_the-remaining-nasone-successor_5195
- 2026-09-10T02:46:30.073220+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:28:59.069889+00:00 (updated-by): Updated: section:notes
- 2026-09-24T02:29:26.534024+00:00 (state-transition): State: pending → superseded
- 2026-09-24T05:20:04.846198+00:00 (state-transition): State: superseded → pending
- 2026-09-24T05:20:31.049912+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
