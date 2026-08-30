# 1003_quantized_cpy_thread_block_fix: Upstream backport: quantized cpy kernels launched with 1 thread per block

**Status:** rejected
**Group:** upstream-fixes
**Plan item:** none

## What it does

Was intended to fix eleven quantized f32<->{q8_0,q4_0,q4_1,q5_0,q5_1}/f32->iq4_nl copy-kernel launches in cpy.cu that computed one block per quantized element/block and launched with a hardcoded 1 thread per block, instead of using CUDA_CPY_BLOCK_SIZE like the rest of the file.

## Why

REJECTED: source commit 69bf6437 (upstream PR #26731) turned out to be an ancestor of this project's own pinned base -- every edit was already-applied at both the b10362 and b10502 pins, so this patch has been a silent no-op since porting, not a functional change. Left in the tree for provenance with STATE=rejected so it is never selected by a default recipe.

## Upstream / provenance

Cherry-picked from merged upstream commit 69bf6437914596fbbc4caf09a7ac16f2acdd1a94 (PR #26731). Turned out to already be ancestral to this project's pinned base.
