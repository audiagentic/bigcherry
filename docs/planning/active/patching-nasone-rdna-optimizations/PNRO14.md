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

# RDNA3.5 D=256 tile FlashAttention occupancy configuration

## Description

RDNA3.5-only (gfx1151) FlashAttention tile occupancy row for D=256, ncols=32: (DKQ 256, DV 256, ncols 32, nthreads 256, occupancy 4, nbatch_fa 64, nbatch_K 64). Disposition TODO -- NOT upstream: b11126 fattn-tile.cuh has only the generic RDNA row GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 3, 64, 128) inside ggml_cuda_fattn_tile_get_config_amd_rdna, used for all RDNA including gfx1151 (earlier 'superseded' disposition was wrong; GPT req_14f299d271894983). gfx1100/gfx1201 must keep the generic row. Hardware-blocked for promotion: no gfx1151 in the fleet, so implement + prove non-selection on gfx1100/gfx1201 now; performance only when gfx1151 is available.

## Steps

1. Create patch package patches/12xx_nro14_rdna35_fa_tile_d256 (next free order in 1262+; check `ls patches`), state untested, kind enhancement, requires [].
2. Edit A (fattn-tile.cuh): insert a new function ggml_cuda_fattn_tile_get_config_amd_rdna3_5 immediately BEFORE the host selector `static __host__ uint32_t ggml_cuda_fattn_tile_get_config(const int DKQ, const int DV, const int ncols, const int cc) {` -- it returns the RDNA3.5 row for (256,256,32) and otherwise delegates to ggml_cuda_fattn_tile_get_config_amd_rdna.
3. Edit B (host selector): replace `        if (GGML_CUDA_CC_IS_RDNA(cc)) {\n            return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);` so RDNA3.5 (GGML_CUDA_CC_IS_RDNA3_5(cc)) calls the new function first.
4. Edit C (device selector): replace `#ifdef RDNA\n    return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);` with an `#if defined(RDNA3_5)` branch returning the new function, else the generic RDNA function (RDNA3_5 is defined in vendors/hip.h:224 for gfx1150/1151).
5. Add an activation marker at the host call site that launches the tile kernel for this config (search fattn-tile.cu for the launch using ggml_cuda_fattn_tile_get_config(...,cc)); emit BIGCHERRY_PATCH_HIT patch=<id> only when GGML_CUDA_CC_IS_RDNA3_5(cc) && DKQ==256 && ncols==32, gated on BIGCHERRY_PATCH_TRACE.
6. Write patch.py with the three anchored edits (exact anchors below), each with a distinctive guard (the new function name / GGML_CUDA_CC_IS_RDNA3_5 line).
7. Offline: patch-lint; patch-rebase-check --focal-overlay <id> --source bigcherry-tuning.
8. Hardware now (Brutus): build gfx1100+gfx1201, run test-backend-ops -o FLASH_ATTN_EXT for head 256 and confirm correctness and that the marker never fires (non-selection controls).
9. Hardware later: on gfx1151 with rocWMMA disabled, D=256 ncols=32 prompt sweep vs generic row; promote only on positive evidence.

## Detailed Solution & Technical Design

Config rows are compile-time constexpr tables (GGML_CUDA_FATTN_TILE_CONFIG_CASE returns early on match), selected by a host function (runtime cc) and a device function (preprocessor arch macros). A dedicated RDNA3.5 function containing only the new row, then falling through to the generic RDNA table, keeps every other shape and every other RDNA arch byte-identical. Both selectors must agree (host decides launch params, device instantiates the kernel), hence edits B and C. gfx1100 (RDNA3_0) and gfx1201 (RDNA4) never define RDNA3_5 and fail GGML_CUDA_CC_IS_RDNA3_5, so they keep (256,256,32,256,3,64,128).

## Code Samples & Guidance

File ggml/src/ggml-cuda/fattn-tile.cuh at b11126.

Edit A (insert_before), anchor:
    static __host__ uint32_t ggml_cuda_fattn_tile_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
text:
    static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_amd_rdna3_5(const int DKQ, const int DV, const int ncols) {
        GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 4,  64,  64)
        return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
    }

guard: ggml_cuda_fattn_tile_get_config_amd_rdna3_5\(const int DKQ

Edit B (replace), anchor (host selector, unique):
        if (GGML_CUDA_CC_IS_RDNA(cc)) {
            return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
text:
        if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
            return ggml_cuda_fattn_tile_get_config_amd_rdna3_5(DKQ, DV, ncols);
        }
        if (GGML_CUDA_CC_IS_RDNA(cc)) {
            return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
guard: if \(GGML_CUDA_CC_IS_RDNA3_5\(cc\)\) \{\n\s+return ggml_cuda_fattn_tile_get_config_amd_rdna3_5

Edit C (replace), anchor (device selector):
    #ifdef RDNA
        return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
text:
    #if defined(RDNA3_5)
        return ggml_cuda_fattn_tile_get_config_amd_rdna3_5(DKQ, DV, ncols);
    #elif defined(RDNA)
        return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
guard: #if defined\(RDNA3_5\)\n\s+return ggml_cuda_fattn_tile_get_config_amd_rdna3_5
(The existing `#else` / `#endif // RDNA` lines after it remain valid.)

patch.toml:
    schema = 1
    id = "12xx_nro14_rdna35_fa_tile_d256"
    order = 12xx
    state = "untested"
    kind = "enhancement"
    backend = "hip"
    plan-ids = ["PNRO14"]
    requires = []
    conflicts = []
    validation-architectures = ["gfx1151"]

patch.py: `from bigcherry.patcher import Edit, FilePatch`; PATCHES = [FilePatch(path="ggml/src/ggml-cuda/fattn-tile.cuh", description=..., edits=(A, B, C))] with mode insert_before / replace / replace as above (follow patches/1204_rd08_q6k_mmvq_vdr2/patch.py conventions).

## Files

ggml/src/ggml-cuda/fattn-tile.cuh (upstream, 3 anchored edits); ggml/src/ggml-cuda/fattn-tile.cu (activation marker at the launch site); patches/12xx_nro14_rdna35_fa_tile_d256/{patch.toml,patch.py,SUMMARY.md,README.md}; tests: test-backend-ops FLASH_ATTN_EXT head 256 cases (existing).

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay <id> --source bigcherry-tuning` CLEAN. Hardware now (Brutus): gfx1100 and gfx1201 builds, test-backend-ops -o FLASH_ATTN_EXT (hs=256) pass, BIGCHERRY_PATCH_TRACE=1 shows NO patch marker (non-selection). Promotion: only on gfx1151 (not in fleet) -- D=256 ncols=32 correctness plus llama-bench prompt sweep vs the generic row, with LDS/VGPR/occupancy evidence.

## Effort & Risk

S / low risk: one constexpr row behind an arch-exclusive selector; risk is only a host/device selector mismatch, covered by edits B+C together. Promotion blocked on gfx1151 hardware.

## Standards

Architecture-scoped evidence; no extrapolation.

## Acceptance Criteria

Correct host/device table selection; no non-RDNA3.5 selection; positive gfx1151 prefill effect with unchanged correctness, otherwise retain deferred status.

## Notes

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
