# Radiance kernels for RDNA3 (RR01)

Goal: standalone [radiance](https://codeberg.org/StillDeadcode/radiance) serving on the RX 7900 XTX cards (gfx1100).
Radiance's device kernel library, libr4d, is built for gfx12 only. This directory holds the work to supply kernels
for gfx11. Plan: `docs/planning/active/build-radiance-rdna3/RR01.md`.

## Where things are

- `kdev/`: the dev harness. Every directory under `kdev/candidates/` is one kernel candidate and builds as its own
  small plugin `k_<directory>`, so candidates of one op run side by side under radiance's checker.
  - `kdev/common/kdev_plugin.h`: plugin exports, the row macro, operand descriptions.
  - `kdev/common/kdev_device.h`: device helpers (bf16 conversion as the reference defines it, row pitch).
  - `kdev/common/kdev_binary.h`: the host side of add / mul, shared so candidates differ only in the kernel.
- `tools/lab/radiance/kdev.sh`: build the candidates, run `rad-kbench` for the given ops on one card, print the
  comparison. `tools/lab/radiance/kdev_report.py` turns the checker's report into the table and a JSON file, and
  compares two runs.
- `tools/lab/radiance/rdna3-survey.sh`: compiles libr4d's units for gfx1100 and reports what fails and why.

## Dev loop

```
VIS=0 SCRIPT kdev-add tools/lab/radiance/kdev.sh @<any build run> <out-dir> add,mul [candidate...]
```

One line per shape, one column per kernel: time per launch, `!` where the kernel disagrees with the reference
(libref, radiance's host implementation of every op), `~` where the timing spread is over 20%. `AGAINST=<earlier
kbench.json>` adds the change in time per kernel. Every buffer the checker hands a kernel has a poisoned margin on
both sides, so an out-of-bounds write fails the case.

## Survey, 2026-10-09 (radiance 1.3.0, commit 89cee7c; ROCm 7.2.4; run `rdna3-survey`)

The engine starts on an XTX (`AQL backend, 1 device(s) [0:gfx1100]`) and loads libr4d without its device code:
"the fat binary has code objects for [gfx1201] and this card runs [gfx1100, gfx11-generic]. Its 201 device kernel(s)
are left out of selection". `rad-kbench` runs on the card; the reference passed 20 of 20 `add` cases.

Compiling libr4d's 66 device units for gfx1100 with only its target filter widened: 28 compile unchanged, 38 fail.

| Cause | First error | Units |
|---|---|---|
| gfx12 matrix-multiply builtins | `__builtin_amdgcn_wmma_*_w32_gfx12` needs `gfx12-insts` (bf16, f16, iu8, fp8) | 21: every quantised and bf16 GEMM except `gemm_bf16_nt_m16`, `gdn_chunk_scan`, `gdn_kkt_solve`, `moe`, `qsa_score_bf16`, `vit_bf16` |
| gfx12 transposed load | `__builtin_amdgcn_global_load_tr_b128_v8i16` in `r4d_common.h:196` | 8: all attention units |
| fp8 conversion instructions | `__builtin_amdgcn_cvt_pk_fp8_f32`, `cvt_pk_f32_fp8` need `fp8-conversion-insts` | 6: `model_bf16` (which holds add, mul, rmsnorm and the other bf16 model ops), `hc_bf16`, `qk_norm_rope_gate`, `quant_act_fp8`, `fused_quant_fp8`, `gdn_recurrent_update` |
| not assembled for this card | `instruction not supported on this GPU` | 4: `ar_gather_2rank`, `ar_oneshot_2rank_exact`, `ar_oneshot_2rank_wht6`, `ar_ln_had_quant_i8` |

(The counts overlap where a unit hits more than one; the table lists each unit under its first error.)

What this means for the port:

- It is not 66 rewrites. Four instruction groups stop the build.
- gfx11 has matrix-multiply instructions for bf16, f16 and iu8 with a different operand layout, and none for fp8.
  The fp8 paths need another compute route (widen to bf16 or f16, or the int8 path).
- The fp8 conversions can be done in software with the same rounding.
- First step after the harness is proven: a compatibility layer for those groups, so the library is correct on
  gfx1100, checked unit by unit against the reference. Then native gfx11 versions of the hot kernels, each compared
  with the layered version in this harness.

Not yet measured: whether the 28 units that compile also compute correctly on gfx1100.

## First harness run, 2026-10-09 (run `kdev-add`, one RX 7900 XTX)

Both first candidates build for gfx1100 (2 s) and pass every `add` case of the fixture bit-exactly against the
reference: 60 checked, 60 passed, worst rel_l2 0; 40 cases ran inside guarded allocations and none wrote outside.
The fixture has no `mul` case, so `mul` was built but not exercised.

Time per launch, microseconds, n = 5120:

| M (rows) | `flat_add` (one thread per element) | `rows_add` (8 elements per thread) |
|---|---|---|
| 1 | 5.3 | 6.1 |
| 16 | 5.5 | 5.9 |
| 64 | 6.9 | 6.9 |
| 128 | 8.4 | 8.8 |
| 256 | 10.7 | 13.3 |
| 512 | 21.1 | 23.1 |

## libr3, first build that loads, 2026-10-09 (run `libr3-b1`, one RX 7900 XTX, commit cc08c2c9)

`libr3/` compiles libr4d's own units for gfx1100 through `r3_compat.h` (software fp8 conversions so far). 28 units
built in 16 s, 34 on `exclude.txt`. The engine loads it on the card: `libr3 0.0.1 kernels 119 kernel(s), 106
schema(s), built for gfx1100` (libr4d itself offers 0 device kernels there, 201 on gfx12).

`rad-kbench --kernels libr3,libref` over radiance's recorded fixture (1098 cases from a Qwen3.8-27B fp8 container):
1597 checks passed, 70 failed, 27 skipped, 7 diverged, 16 capped; worst rel_l2 4.4e-04; 542 cases ran inside guarded
allocations and none wrote outside. 56 of the 209 declared rows were reached by this fixture.

Device kernels that pass every case they were given (23): `add_bf16`, `rmsnorm_bf16` (40 cases), `rope_bf16` (37),
`rope_table_f32`, `silu_mul_bf16`, `gather_rows_bf16`, `kv_store_bf16`, `kv_store_fp8`, `quant_act_fp8` (44 cases,
bit-exact: this is the software fp8 conversion), `gdn_gated_rmsnorm_h128_bf16`, `dflash_select_bf16`, `rowtopk_bf16`
and the whole sampler chain (`sample_argmax`, `dry`, `mask`, `minp`, `penalties`, `pick`, `temp`, `topk`, `topp`,
`typical`, `xtc`).

The 70 failures are three kernels refusing, none computing a wrong number:

| Kernel | Cases | What it says | Likely cause (not yet confirmed) |
|---|---|---|---|
| `gemm_bf16_nt_m16` | 30 | `unsupported` | calls into a unit on `exclude.txt`; the stub answers |
| `cast_bf16_bf16` | 20 | `device error` | the launch itself is refused on gfx11 |
| `gdn_conv_update_w4_h128_bf16` | 20 | `shape this kernel does not serve` | to be read |

Skipped: `logits_gemm` and `embed_lookup` cases need about 7 GiB against a 2.65 GiB budget (half of the free VRAM
while other jobs hold the card); rerun with `--max-bytes` when the card is free.

96 of libr3's rows were not reached by this fixture; the routed-MoE fixture (`kernels_moe.rkb`) and a bf16 model
fixture will reach more.

## libr3, second check, 2026-10-09 (run `libr3-b2`, commit e2ac3976)

All five "instruction not supported" failures were one instruction, gfx12's `s_wait_storecnt`. libr3 now builds from
a copy of libr4d's sources in the build tree with one substitution (`rewrites.txt`: `s_waitcnt_vscnt null, 0x0`),
which matched 8 places. 33 units built, 29 left out; the engine sees 123 device kernels.

`rad-kbench`: 1657 checks passed, 50 failed, 27 skipped; 602 cases in guarded allocations, none wrote outside.
26 device kernels pass every case: the 23 above plus `cast_bf16_bf16` (its 2D copy is in a recovered unit),
`gated_quant_fp8` and `rmsnorm_quant_fp8` (worst rel_l2 2.3e-03, inside the checker's tolerance).

Two kernels still refuse: `gemm_bf16_nt_m16` (30 cases, `unsupported`: it reaches a unit that needs the gfx12 WMMA
builtins) and `gdn_conv_update_w4_h128_bf16` (20 cases, `shape this kernel does not serve`, not yet read).

Timings of the kernels shared with the first check are unchanged (geometric means 0.94 to 1.08 of the earlier run).

## First kdev run

`flat_add` is faster or equal at every shape (geometric mean 7.2 us against 7.8 us). Up to M = 32 the time is the
launch itself, about 5.5 us. This run proves the loop; the two kernels are deliberately simple.
