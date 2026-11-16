# rad0040_kv_layout_table

**Status:** untested
**Plan item:** VTP44

Scope: the packed KV layouts as one table, with no change in behaviour.

## Summary

`abi/rad_builder.h` gets `RAD_KV_LAYOUTS`, one line per packed `--kv-cache-dtype` layout (enum id, name, row bytes per
16 dims, scale group). The `RAD_KVQ_*` enum, `rad_kvq_name`, `rad_kvq_from_name` and `rad_kvq_row16` are generated from it, and
the flag parser (accepted names, its refusal text and its help line), the core's name-to-layout map and the Qwen3.5 FP8
plugin's name and row size read it instead of keeping their own lists. libr4d's `bf16_fp8` and `bf16_i8` attention rows
(constraints, names, descriptions) are generated from one tuple list in `r4d_rows.cpp`, with a compile-time check that
each tuple names a table layout. A new layout or geometry of these is a line in a list, not a copied row.

## Outcome

Untested. The rows object (`r4d_rows.cpp` compiled on the host) is byte-identical to the one before the patch: the same
section bytes and relocations. See README.md.

## Detail

See README.md.
