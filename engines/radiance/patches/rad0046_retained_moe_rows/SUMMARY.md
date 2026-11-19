# rad0046_retained_moe_rows

**Status:** untested
**Plan item:** RUM03

## What it does

Restores the `had_quant_act_fp8` and `moe_gemm_fp8a8` rows on the retained implementations. The grouped GEMM keeps 64-column fallback tiling unless every expert class and part aligns to the 128-column block-scale grid. The shared pure width predicate lives in `libr4d/r4d_moe_block_fp8_geometry.h` so the launcher and card-free boundary fixture exercise the same rule.

## Why

The Flash-Next and gfx1201 helper paths resolve these exact row names and their existing E4M3 code and scale planes. Unequal class widths can cross a block-scale boundary in a wide tile.

## Upstream

Radiance 2.1.0 candidate `3deabdaeca17b58bad90d332f4b204cf7bb8a10e` removed the row declarations while retaining their launch templates, constraints and operand descriptions. Candidate source replay passed. Plugin build, hardware selection and numerical qualification remain pending.
