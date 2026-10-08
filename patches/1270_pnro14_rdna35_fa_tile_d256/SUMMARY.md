# 1270_pnro14_rdna35_fa_tile_d256

**Status:** untested, not promoted. **Owner:** PNRO14. **Disposition:** BCOP74 (2026-10-09).

This is the existing RDNA3.5 D=256/ncols32 tile configuration experiment: 256 threads, occupancy target 4, nbatch_fa 64, nbatch_K 64 versus native occupancy 3, nbatch_K 128. It changes only tile configuration after the normal attention-kernel selection. Source analysis shows standard aligned masked GQA prefill generally selects MMA before the 32-column tile; an actual TILE+ncols32 trace is required before benchmarking.

**Architecture safety:** current selectors cover gfx1150/1151/1152/1153, but validation-architectures lists only gfx1151. Do not promote until both selectors are narrowed or all admitted architectures are qualified. Existing once-only marker is not per-launch evidence. PNRO19 mechanics failures were fixed by f4228540b7; no new patch is required.

**Gate:** real gfx1151 hardware, tile32 attribution >=5% E2E, backend-reference correctness and repeated ABBA CI95-low >=3% E2E prefill gain with <=1% decode/control regression. No BigCherry gain measured. AMD PR #41's pp1024 +0.3% was a single-run external result.

References: PNRO14; PNRO19; BCOP74; https://github.com/AMD-Ecosystem/llama.cpp/pull/41 .
