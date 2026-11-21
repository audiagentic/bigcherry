# rad0042_kv_b16_selftest

**Status:** diagnostic
**Plan item:** VTP44

Scope: r4d_selftest cases for the `bf16_fp8` and `bf16_i8` KV layouts on Flash-Next's attention geometry (rad0041).

## Summary

`libr4d/r4d_selftest.cpp`: the Flash-Next QSA numeric cases (gqa 12 and gqa 6 with 4-token pages) take the two layouts, with a ragged last prefill tile and the decode at every wave count (1 to 5 rows at gqa 12, 4 and 8 rows at gqa 6); the split-KV prefill and mixed-step tables gain both geometries for both layouts; the fused gate+quantiser byte-identity case (`attn_decode_gq_h256_gqa12_*` against the decode and `gate_quant_fp8` pair) builds a bf16-K cache with finite E4M3 or int8 V bytes and f16 group scales.

## Outcome

The cases compile in a gfx1201 build and pass libr3's rewrite check. 2026-11-18: run on a gfx1100 card (one 7900 XTX, libr3's selftest, each case alone): all 21 cases of the two layouts pass, including `bf16_i8_rows` (after the inventory fix below) and the gqa6/block-4 fused-decode rows (docs/evidence/2026-11-16-vtp44-generic-kv-layouts/rad0042-selftest/). The gfx1201 device run was not repeated. Qualified as a diagnostic tool on 2026-11-20 (record in `evidence/promotion.json`): the overlay is test text only, so it is classified diagnostic rather than validated.

## Detail

See README.md.

## 2026-11-18 additions (GPT review of the KV patches; found by a gfx1100 device selftest)
- The `bf16_i8_rows` inventory case (rad0038) listed only the four gqa 6 / 16-token-page rows and so rejected the six rows rad0041 added (gqa 12, and gqa 6 at four ranks, over 4-token pages); on a gfx1100 device it failed. It now lists all ten rows, each with the gqa and block size it must pin, and no other row may take `bf16_i8`.
- The fused gate+quantiser fold case ran at gqa 12 only, so the two gqa 6 / 4-token-page fused rows (`attn_decode_gq_h256_gqa6_bs4_<layout>`) had no execution test. `attn_gq_fold_case` takes a geometry flag, and the packed layouts run at both, at 3 and 24 rows, against their unfused pair.

## Radiance 2.2 host replay

The diagnostic patch now depends on the ABI44 B16/V4 implementation ending at rad0053. The packed cases use opaque byte tensors and preserve the Flash-Next QSA numerical, split, mixed-step and gate fold matrix. The `bf16_i8_rows` case checks the fourteen B16 int8 registrations in this base, including the additional fused store and Hadamard rows.

The patch applies and reapplies to the composed Radiance 2.2.0 source at `b6b8751cccb0e834a13ae9e5afca824ee9a487aa`. The patched C++20 selftest passes syntax checking and `bc_kv_layout_diagnostic_test` passes under GCC 14 with `NDEBUG` enabled. The historical gfx1100 results above cover the earlier base. Device execution on Radiance 2.2 remains pending.
