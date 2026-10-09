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

Not yet measured: whether the 28 units that compile also compute correctly on gfx1100, and any timing.
