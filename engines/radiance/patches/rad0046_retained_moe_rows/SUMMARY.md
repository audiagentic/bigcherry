# rad0046_retained_moe_rows

**Status:** untested
**Plan item:** RUM03

## What it does

Restores the `had_quant_act_fp8` and `moe_gemm_fp8a8` rows on the retained implementations. The grouped GEMM keeps 64-column fallback tiling unless every expert class and part aligns to the 128-column block-scale grid. The shared pure width predicate lives in `libr4d/r4d_moe_block_fp8_geometry.h` so the launcher and card-free boundary fixture exercise the same rule.

## Why

The Flash-Next and gfx1201 helper paths resolve these exact row names and their existing E4M3 code and scale planes. Unequal class widths can cross a block-scale boundary in a wide tile.

## Upstream

The row declarations are missing from upstream Radiance 2.2.0 (`b6b8751cccb0e834a13ae9e5afca824ee9a487aa`), which retains the launch templates, constraints and operand descriptions. Exact-source apply and byte-identical replay passed. The composed ABI44 libraries expose both rows. A gfx1201 probe resolved the quantizer and checked the grouped GEMM against a numerical reference at both 64- and 128-column tile widths. The production promotion and matched deployment regression gates remain open.
