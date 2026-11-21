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

## Review findings (GPT code review, `docs/evidence/2026-11-16-vtp44-generic-kv-layouts/code-review-a.md`)

- Tuple fields that are macros: names and text are built with `R4D_XSTR` (expand, then stringify) and `R4D_TUPLE_CAT4`
  (expand, then paste), so a named head_dim or geometry gives its value, never the macro's name. Checked on the real file: a copy
  whose first tuple uses macros for head_dim, gqa, block, q_len and geometry compiles to the identical object.
- The tuple list is for the `attn_paged` prefill and decode pair only. The fused gate+quantiser decode
  (`attn_paged_gate_quant`) is another op kind with its own constraints and entry points: patch B gives it its own tuple
  list and generator, not extra fields here. A geometry exists only when its tuple and kernel instantiation do.
- The C++ standard is 17 or 20 across the build (`compile_commands.json`: gnu++17 and gnu++20), so the constexpr loop is valid.

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

### Promotion record

Tier: profile-evidence

Model: Qwen3.8-27B w8a8 (.rad), qwen27b-tp3-int8-556-long-bf16 (TP3 over two RX 7900 XTX and an R9700, --kv-cache-dtype bf16_i8 and bf16_fp8, up to 262,144-token context)
Model: Qwen3.8-Flash-Next hcbf16 (.rad), two RX 7900 XTX at TP2 with exact wire, experts tiered through the host pool, no helper, no speculation (--kv-cache-dtype bf16_i8 and bf16_fp8)
Mechanics: r4d_selftest on the R9700 (gfx1201) with rad0040, rad0041 and rad0042 applied: 0 failures, 248 skipped (baseline 246; the two extra are the gqa6 block-4 fused rows with no smoke case, as for fp8_v4 and k6_v4), every baseline ok line still passes and 274 new rows pass, including fused gqa12 byte-identity (docs/evidence/2026-11-16-vtp44-generic-kv-layouts/DEVICE-AB.md); libr3 and libr4d compile clean for gfx1100 and gfx1201 with no spill above the existing worst case for bf16_i8 (device-compile-b.md); arch_test, config and bringup tests, patch-lint and the docs catalog tests passed on the patch PRs (#1012, #1013); the libr3 gfx1100 device selftest was not run, the gqa12 and gqa6 rows ran on gfx1100 in Flash-Next serving on the two XTX (FLASHNEXT-AB.md).
Activation: `--kv-cache-dtype bf16_i8` or `bf16_fp8` selects the packed layout from the one layout table: the engine reports the layout and sizes pages from it (Flash-Next attention page 36.75 and 36.00 KiB against 48.00 KiB for bf16, FLASHNEXT-AB.md; 27B 818,896 and 835,952 tokens against 624,048 for bf16, VTP43.md); an unsupported (layout, head_dim, gqa) refuses at startup naming the tuple; the default caches are selected exactly as before.
Identity: the default caches are unchanged: against references the stock build recorded, the patched build scores Flash-Next bf16 6.4e-11 and fp8 3.3e-11 mean KLD with top-1 100% over 4,161 positions (IDENTITY.md), and 27B bf16 6.1e-11 and fp8 4.7e-11 on the short corpus (42,761 positions) and bf16 4.8e-11 on the long one (276,900), top-1 100% (EQUALITY-27B.md); at the tool's float floor, not bit-exact (two binaries); the packed caches differ from bf16 by design and are measured in the A/B line.
A/B: 27B against the stock build's bf16 reference (EQUALITY-27B.md, VTP43.md): long-corpus mean KLD bf16_i8 0.002685 and bf16_fp8 0.003228 against fp8 0.004947 (bar 0.0027 met by bf16_i8), short 0.001411 and 0.001922 against fp8 0.002235; capacity 1.31x and 1.34x bf16; decode at 1, 4 and 8 streams and TTFT within 3% of fp8 (VTP43.md). Flash-Next (FLASHNEXT-AB.md, single runs, two XTX, no helper, no speculation): decode and TTFT within about 1% of bf16 for bf16_i8, bf16_fp8 +6.7% on one-stream decode and 5.9% slower 48K TTFT; capacity 1.283x (bf16_i8) and 1.307x (bf16_fp8) of bf16 because the QSA block-key cache is not quantised; needle probe at 48K tokens 9 of 9 and 8 of 9 against 8 of 9 for bf16; Flash-Next KLD against bf16 sits inside the model's arithmetic-noise band (re-chunking the bf16 cache alone gives the same) so it cannot judge KV error (FLASHNEXT-KLD.md); the R9700 helper arm was not run.
