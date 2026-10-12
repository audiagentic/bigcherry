# 1266_rd05_wmma_fa_tileq_sync

**Status:** untested
**Plan item:** PRBE110

## What it does

Fresh untested PRBE110 extraction of RD05 only: WMMA flash-attention `tile_Q` reuse synchronization plus activation instrumentation. RD06/RD07 are excluded. Validate under `PRBE110-RD05-WMMA-TILEQ-SYNC`; do not reuse rejected 1203 evidence.

## Current-pin disposition (2026-10-12)

The package remains **registry-untested/default-off**, not a production correctness fix. Four historical b11126 gfx1201 sessions recorded activation and full-vocabulary parity, but no qualifying performance improvement or reproduced tile_Q race. On the selected Radiance development branch, `config/recipes.toml` still pins b11474; main pins b11552. The latter includes upstream #30103's `np > 1 || nbatch_combine != DKQ/2` guard, so `patch.py`'s literal `if (np > 1)` replacement is stale and must fail closed. Do not force-apply, delete the guard, or transplant the old k00 condition. The inter-`kbc` synchronization hypothesis remains unproven; PRBE02 owns the bounded reproduction/retirement gate. PRBE110 owns only the independent Q6_K/1267 optimisation. No new build, GPU run or speedup is claimed.
