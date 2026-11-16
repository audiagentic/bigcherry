# rad0040_kv_layout_table

## Summary

VTP44 patch A. rad0037 and rad0038 added the `bf16_fp8` and `bf16_i8` KV layouts by hand-writing names in four places
and one row per (layout, op) in libr4d. This patch makes the layouts data, so the next geometry (Flash-Next, VTP44 patch B)
and the next layout are additions to a list. It changes no behaviour: the same names, sizes, rows and messages.

| piece | where |
|---|---|
| `RAD_KV_LAYOUTS(X)`: `X(id, name, row16, group)`; the enum, `rad_kvq_name`, `rad_kvq_from_name`, `rad_kvq_row16`, `RAD_KV_LAYOUT_NAMES` | `abi/rad_builder.h` |
| flag validation, its refusal text and the help line | `core/config.cpp` |
| name to layout (`bc_kv_quant_of`) | `core/rad_core.h` |
| declared layout name and row size | `arch/qwen35_fp8/qwen35_fp8.cpp` |
| comments that listed the names | `core/engine_bringup.cpp`, `core/kvstream/bringup.cpp` |
| the bf16-keyed attention rows: `R4D_B16_ATTN_TUPLES` generates the constraint arrays and the prefill and decode rows; `static_assert` that a tuple's `kv_dtype` is the table's name | `libr4d/r4d_rows.cpp` |
| table test | `tests/config_test.cpp` |

What stays per geometry, on purpose: the kernel instantiation a tuple names (tile shape, LDS, option bits) and the
`gqa == 6` check in the plugin. The check moves into a shared helper when a second plugin needs it (patch B); one
plugin's check is not a reason to widen the ABI now. The hand-written `fp8_v4` and `k6_v4` rows and the upstream rows are
untouched.

## Proof of no change

`libr4d/r4d_rows.cpp` compiled on the host (`g++ -c`, no device code) before and after: `objdump -s` (every section's
bytes) and `objdump -r` (every relocation, so every row's function pointers) are identical; only the local names of
four static constraint arrays differ in the symbol table, at the same addresses. Warning counts are unchanged (g++
3459, clang++ 1848). `config_test` passes including a table test; `arch_test` and `mem_test` have the same failures as before the
patch (2 and 23, from patches the scratch tree lacks). The libr3 rewrite check (`r3_rewrite.py --check`) passes.
A negative control (one row size changed) makes the table test fail.

## Outcome

Untested; the engine-level selftests and a KLD equality run follow once the device baseline exists (see
`docs/evidence/2026-11-16-vtp44-generic-kv-layouts/BASELINE-STATUS.md`).

## Detail
