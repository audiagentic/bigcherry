# rad0041_kv_b16_flashnext

**Status:** untested
**Plan item:** VTP44

Scope: the `bf16_fp8` and `bf16_i8` KV layouts on Flash-Next's attention geometry (head_dim 256; gqa 12, and gqa 6 with 4-token pages).

## Summary

libr4d's gqa8-body units (`r4d_attn_paged_h256_gqa12.hip`, `r4d_attn_paged_h256_gqa6_bs4.hip`) gain the two layouts' prefill, decode and fused gate+quantiser entry points behind a new `R4D_ATTN_B16` flag, built from the existing gqa6 templates' `O_KB16V8` / `O_KB16I8` forms. One tuple list in `r4d_rows.cpp` (`R4D_B16_QSA_TUPLES`) generates their twelve rows (decode causal or not), with compile-time checks that a tuple's `kv_dtype` is the layout table's name and its (layout, head_dim, gqa) is in `RAD_KV_GEOMETRIES`. `abi/rad_builder.h` gets `RAD_KV_GEOMETRIES` and `rad_kvq_geometry_ok`; the Qwen4 expert FP8 plugin reads the layout and geometry tables for its name, row size and refusal, and its probe offers every layout the table lists for the model's geometry.

## Outcome

Untested. Compiles for gfx1201 (no spill above the existing worst case for `bf16_i8`; `bf16_fp8` single-wave decode spills 45 VGPRs, to be measured) and, through libr3's rewrites, gfx1100. Device selftests and Flash-Next KLD / speed are pending. See README.md.

## Detail

See README.md.
