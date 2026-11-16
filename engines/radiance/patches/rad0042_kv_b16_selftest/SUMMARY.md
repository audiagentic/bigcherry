# rad0042_kv_b16_selftest

**Status:** untested
**Plan item:** VTP44

Scope: r4d_selftest cases for the `bf16_fp8` and `bf16_i8` KV layouts on Flash-Next's attention geometry (rad0041).

## Summary

`libr4d/r4d_selftest.cpp`: the Flash-Next QSA numeric cases (gqa 12 and gqa 6 with 4-token pages) take the two layouts, with a ragged last prefill tile and the decode at every wave count (1 to 5 rows at gqa 12, 4 and 8 rows at gqa 6); the split-KV prefill and mixed-step tables gain both geometries for both layouts; the fused gate+quantiser byte-identity case (`attn_decode_gq_h256_gqa12_*` against the decode and `gate_quant_fp8` pair) builds a bf16-K cache with finite E4M3 or int8 V bytes and f16 group scales.

## Outcome

Untested: the cases compile in a gfx1201 build and pass libr3's rewrite check; they have not run on a device. See README.md.

## Detail

See README.md.
