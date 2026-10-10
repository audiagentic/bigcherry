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

## libr3, third check: GEMM and attention pass, 2026-10-09 (run `libr3-b3`, commit ab594c0f)

`r3_compat.h` now emulates the gfx12 bf16 and f16 WMMA instructions and the gfx12 transposed load on gfx11 (layouts
from BigCherry patch 1253, which probed them on these cards). 51 units build in 29 s, 11 left out; the engine sees
176 device kernels.

`rad-kbench --kernels libr3,libref`: **1869 checks passed, 0 failed**, 27 skipped, 7 diverged, 32 capped; worst
rel_l2 4.3e-03; 814 cases ran inside guarded allocations and none wrote outside. 34 device kernels were reached and
all pass. New since the second check:

| Kernel | Cases | Worst rel_l2 | Geometric-mean time |
|---|---|---|---|
| `gemm_bf16_nt_m16` | 30 | 4.5e-04 | 16.5 us |
| `gemm_bf16_nt_m64` | 26 | 1.1e-04 | 19.0 us |
| `attn_decode_h128_gqa4_bf16kv` | 12 | 2.3e-03 | 33.2 us |
| `attn_decode_h256_gqa6_fp8kv` | 16 | 2.3e-03 | 191.7 us |
| `attn_prefill_h256_gqa6_fp8kv` | 8 | 1.9e-03 | 1295.9 us |
| `gdn_chunk_scan_k128_v128_c64_bf16` | 10 | 4.3e-03 | 78.8 us |
| `gdn_kkt_solve_k128_c64_bf16` | 10 | 8.1e-05 | 15.3 us |
| `gdn_conv_update_w4_h128_bf16` | 10 | 4.5e-05 | 32.6 us |

The two kernels that refused before (`gemm_bf16_nt_m16`, `gdn_conv_update_w4_h128_bf16`) pass now that the units
they reach are built. The emulation costs 12 cross-lane moves per WMMA call and 16 per transposed load, so these
times are a correctness baseline, not a speed claim; no gfx12 time on the same shapes has been taken for comparison.

Still left out (11 units): the int8, 4-bit and fp8 WMMA GEMMs and the routed MoE unit. 142 of libr3's rows are not
reached by this fixture (recorded from a Qwen3.8-27B fp8 container, whose GEMMs are the fp8 ones still excluded).

## libr3, fourth check: every unit builds, the fp8 GEMMs pass, 2026-10-09 (run `libr3-b4`, commit 8f0cf127)

`r3_compat.h` now also emulates the gfx12 int8, 4-bit and fp8 WMMA instructions and maps the gfx12 barrier
builtins to `s_barrier`. All 62 units build in 260 s, none left out; the engine sees 201 device kernels.

`rad-kbench --kernels libr3,libref`: **2083 checks passed, 0 failed**, 27 skipped, 7 diverged, 32 capped; worst
rel_l2 4.3e-03; 1028 cases ran inside guarded allocations and none wrote outside. 37 device kernels were reached and
all pass. New since the third check, all through the fp8 WMMA emulation (fp8 operands widened to bf16):

| Kernel | Cases | Worst rel_l2 | Geometric-mean time |
|---|---|---|---|
| `gemm_fp8a8_nt_m16` | 72 | 9.9e-05 | 808.3 us |
| `gemm_fp8a8_gated_nt_m16` | 7 | 2.4e-07 | 1502.1 us |
| `gemm_fp8a8_tiled` | 28 | 1.7e-04 | 9410.9 us |

So the main GEMMs of the recorded fp8 model are correct on gfx1100. They are slow: the fp8 emulation converts every
operand in software on each call, and these are the first kernels to replace with native gfx11 versions.

Not tested: the fixture holds no case for the int8 and 4-bit GEMMs (`gemm_w8a8_*`, `gemm_w4a8_*`, `gemm_w2a8_nt`,
`gemm_mxfp4a8_*`) or the MoE kernels (`moe_*`), so their emulation compiles but its byte and nibble layouts are
still assumed. They need a fixture recorded from an MXFP4 or int-quantised container. 164 of libr3's rows are not
reached by this fixture.

## libr11 beside libr3, 2026-10-09 (run `libr11-c1`, `tools/lab/radiance/libr11-compare.sh`)

The other session's hand-written gfx11 plugin (`tools/lab/radiance-gfx1100/libr11`) on the same XTX and the same
fixture cases as libr3. 630 checks, 0 failed, for both libraries.

| Op | libr3 (compatibility layer) | libr11 (native gfx11) | Reading |
|---|---|---|---|
| `add` bf16, 10 shapes | 8.4 us geometric mean | 6.4 us | libr11's packed add is faster up to M=256 (5.3 against 7.5 us at M=1); libr3 is faster at M=512 (13.9 against 15.3 us) |
| `gemm_nt` bf16, M<=16, 30 shapes | 16.0 us (9.6 to 40.0) | 267.9 us (251.8 to 284.0) | libr3 is 7 to 27 times faster |

libr11 has no `mul` row in this fixture and no GEMM above M=16. Its own `r11_gemm_smoke` ctest exits non-zero
although every case it prints passes; not looked into.

What to take: the packed add form for small M (as a kdev candidate against `binary_flat`). Not the GEMM: libr4d's
kernel through the emulated WMMA is already far ahead of it, so the native GEMM work should start from libr4d's
tiling with gfx11 fragments, not from libr11's.

## libr3 on an XTX over libr4d on the R9700, 2026-10-09 (runs `libr3-b4`, `r4d-base2`)

`tools/lab/radiance/r4d-baseline.sh` runs radiance's own libr4d on the R9700 over the same fixture (2083 checks, 0
failed) and prints libr3's time over it for every shared case. Different cards, so each ratio is card plus
compatibility layer together. Geometric mean of time on the XTX / time on the R9700 (above 1 = the XTX is slower):

| Group | Kernels | Ratio |
|---|---|---|
| fp8 GEMM (fp8 WMMA emulated through bf16) | `gemm_fp8a8_tiled` 33.2, `gemm_fp8a8_nt_m16` 12.2, `gemm_fp8a8_gated_nt_m16` 4.7 | 5 to 33 |
| fp8-KV attention | `attn_decode_h256_gqa6_fp8kv` 11.1, `attn_prefill_h256_gqa6_fp8kv` 6.5 | 6 to 11 |
| bf16 WMMA kernels | `gdn_chunk_scan` 3.1, `attn_decode_h128_gqa4_bf16kv` 2.4, `gemm_bf16_nt_m16` 2.1, `gemm_bf16_nt_m64` 1.8 | 2 to 3 |
| no WMMA (norms, rope, casts, samplers, quant, KV store) | 28 kernels | 0.57 to 1.46, most within 10% of 1 |

So the layer costs nothing where there is no WMMA, about 2 to 3 times on bf16 WMMA, and 5 to 33 times wherever fp8
is involved. Native gfx11 kernels are needed first for the fp8 GEMMs and fp8-KV attention; a bf16 or f16 container
avoids the fp8 cost without any new kernel.

## First model served on one XTX, 2026-10-10 (run `r3-serve1`, `tools/lab/radiance/libr3-serve-smoke.sh`)

radiance 1.3.0 with `--kernels libr3,libref --tp 1` on one RX 7900 XTX serves the MXFP4 Qwen3.8-27B container
(19 GB file, 14.15 GiB of weights on the card, token embedding in host memory, bf16 KV, context 8192, drafter off).
Healthy 10 s after start; 16.60 GiB allocated, 6.44 GiB of the card left free.

| Request | Reply | Speed |
|---|---|---|
| completion, "The capital of France is", 64 tokens, greedy | " Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome. ..." | 7.0 tok/s, first token 0.18 s |
| chat, "numbers one to ten", thinking off | "1, 2, 3, 4, 5, 6, 7, 8, 9, 10" | 7.7 tok/s, first token 0.46 s |

The text is right, which is the first evidence for the emulated int8 and 4-bit WMMA layouts: this container's GEMMs
are the `gemm_mxfp4a8_*` kernels the fixture never reached. It is evidence from coherent output, not a per-kernel
check against the reference; a fixture recorded from this container is still needed for that. Server exit 0 on SIGINT.

7 tok/s is the emulation's cost, not the card's: every 4-bit and int8 WMMA call goes through software operand
conversion and cross-lane moves. Not measured here: prefill speed on a long prompt, the drafter, more than one card.

## Native gfx11 forms, first kernel: the MXFP4 decode GEMM, 2026-10-10

`libr3/native/<libr4d source>.rw` holds block rules that replace a kernel's own lines with a native gfx11 form when
the sources are copied into the build tree (`r3_rewrite.py`; a block that no longer matches fails the configure
step; `-DR3_NATIVE=OFF`, or `R3_NATIVE=OFF` for `libr3-build.sh`, builds without them). The helpers are in
`r3_compat.h` under "native gfx11 forms".

Same serve test as above (MXFP4 Qwen3.8-27B, one XTX, drafter off, greedy), one run each:

| Build | Decode | `gemm_nt_q` N=34816 K=5120 (FFN gate/up), per call |
|---|---|---|
| all emulated (`r3-serve1`, 64 tokens) | 7.0 / 7.7 tok/s | not profiled |
| native loop, fp8 widened in the K loop (`r3-serve2`, commit d30d2d51) | 8.9 / 8.8 tok/s | 780 us |
| one-fragment form stages bf16 (`r3-serve4`, commit d105d8f7) | 18.9 / 18.8 tok/s | 332 us |
| weights from 4-bit codes to bf16 by table lookup (`r3-serve6`, commit ecddc2bf) | 28.4 / 28.4 tok/s | 208 us |

The text is the same in all three. What the kernel does on gfx12 is an fp8 x fp8 WMMA on E4M3 staged in shared
memory (the MXFP4 weight is unpacked to E4M3 with its block exponent folded in). Native: every lane reads its row's
16 values, the gfx11 bf16 WMMA takes the accumulator as its C operand, and the accumulator is moved to the gfx12
layout once before the unchanged epilogue. In the one-fragment form (M <= 16) the codes are widened to bf16 once, as
they are staged (39 KiB of shared memory a block at a 128-wide slab).

Per-op profile of a decode run with the last build (`--profile-ops`, run `r3-serve5`; the engine warns a profiled
run is not comparable with an unprofiled one, so read it as shares): the MXFP4 GEMMs are still 88% of device time
(38% gate/up, 20% down, 16% the 16384-wide projection, 11% the K=6144 one, 4% the 12288-wide one); the bf16 logits
GEMM is 2.8%, everything else under 2.5% each. With the first native build they were 95%.

Two things tried after the table-lookup build that did not help (2026-10-10):

- **A vector kernel for one token** (RR07, `r3_mxfp4_gemv.h`, opt-in with `R3_GEMV=1`): no matrix instruction, the
  weight decoded as it streams. Its second form took 418 us for the gate/up GEMM against 208 us for the matrix form
  (15.5 tok/s against 28.4), so the 16-row fragment spent on one token is not what limits decode here; the 4-bit
  unpack and the per-block staging, which both forms share, are. It is also not correct: all three forms (bf16 dot
  product, f16 dot product, plain f32 math) fail the selftest's four M = 1 cases with the same error (rel_l2 1.519
  at N=272 K=640), which points at how it reads or pairs its operands; not found. Parked.
- **The split, slab and load hint** (`R3_MXD_KS`, `R3_MXD_BK`, `R3_MXD_NT`; libr4d's rule was measured on gfx12):
  one build, eight settings, decode tok/s: rule 28.5, slab 64 26.5, split 2 27.6, split 4 28.1, split 4 with slab
  64 27.8, split 8 28.8, non-temporal off 27.9, on 28.5. Flat within 8%; the rule stays.

Numeric check (RR06): radiance's own kernel selftest carries a host reference for the MXFP4 GEMMs. Its binary runs
only on gfx1201, so libr3 builds a copy for gfx11 (`r3_selftest`, `tools/lab/radiance/libr3-selftest.sh`). On the
R9700 radiance's kernels pass 88 of 88 cases (74 decode, 4 nt_m64, 10 tiled; run `r4d-self1`); on the XTX libr3
with the table-lookup native form passes the same 88 (run `r3-self3`, commit da3dcd0d).

rad-kbench itself cannot verify this kernel: with this container it skips every MXFP4 GEMM case (its reference
reader wants half the elements the 4-bit plane holds), so the fixture recorded from it
(`tools/lab/radiance/record-fixture.sh`, 559 cases, all passing through libr3) has none. The evidence is the
unchanged greedy text. For scale, radiance on the R9700 runs this model at 37-38 tok/s without the drafter.

## What hipfire does on gfx1100 (read 2026-10-10)

`github.com/Kaden-Schutt/hipfire` (Apache-2.0 from v0.3.0; some files MIT by SPDX), commit 0f999cb, cloned to
`/mnt/data/bigcherry-work/engines/hipfire` on Brutus. A Rust engine with its own HIP kernels; the RX 7900 XTX is
its primary target and it has 60-odd kernels written for gfx1100 by name. Its published Qwen3.8-27B (MQ4) numbers,
self-measured, median of 3:

| | 7900 XTX | R9700 |
|---|---|---|
| prefill, 8192 tokens | 3,021.5 tok/s | 5,166.1 tok/s |
| native MTP decode | 87.5 tok/s | 68.0 tok/s |

So on kernels written for each card the XTX is 29% faster at decode (memory-bound; `docs/BENCHMARKS.md` puts its
27B decode at about 650 GiB/s of the card's 960 GB/s) and the R9700 is 71% faster at prefill (matrix-compute-bound).
The XTX prefill figure is about what radiance reaches on the R9700 here (3,090 to 3,150 tok/s).

Techniques, by kernel, that apply to libr3:

- **Decode is a GEMV, not a WMMA.** `gemv_hfp4g32.gfx1100.hip`, `gemv_hfq4g256.gfx1100.hip`: one weight row a
  32-thread block, the 4-bit codes decoded through a 16-entry table, four interleaved accumulators, no shared-memory
  staging of weights, no matrix instruction. radiance's decode GEMM spends a 16-row fragment on one token; on the
  XTX that is compute the card does not have to spare, and it is why libr3 is at 19 tok/s where the memory rate
  allows about 45.
- **Prefill is one wave a block with the lane owning a weight row.** `gemm_mq4g256v2_residual_wmma_gfx11_bt.hip`:
  a lane reads its row's 16 codes (8 bytes), dequantises them to f16 in registers once per K tile, and reuses that
  fragment across B = 4, 6 or 8 batch tiles with independent accumulators. The activation fragment is a plain
  32-byte load of 16 f16 values of one token. That is the gfx11 fragment used as it is: no conversion in the loop,
  no shared memory, no cross-lane move. The accumulator is written out in the gfx11 interleaved layout directly.
- **Activations are f16 (or int8), never fp8.** gfx11 has no fp8 matrix instruction; hipfire's KV default on gfx1100
  is Q8 for the same reason.
- **Shared memory for the activations when the weight is wide.**
  `gemm_hfq4g256_residual_wmma_gfx1100_muse_lds_g256.hip`: 8 waves share one staged group of X (96 tokens x 256 K,
  49 KiB, laid out tile-major to keep bank conflicts at four-way), cutting X traffic about 16 times.
- **An int8 route.** `gemm_mq4_packed.gfx1100.hip`: activations quantised to int8 in groups of 32 and the 4-bit
  weights multiplied with the integer WMMA; its checkpoint note (`docs/perf-checkpoints/2026-09-29-...`) reports
  +22% prefill over the f16 route and +34% with wider prefill chunks, on one fixture.

## References for the remaining port (found 2026-10-09; read before writing kernels)

The 29 excluded units need the gfx12 WMMA builtins or the gfx12 transposed load. These exist to adapt from, so none
of it should be written from nothing:

- **A gfx1100 deployment of the older radiance, of limited use here.** `github.com/mkadrlik/vllm-radiance-p2p`:
  radiance 0.5.7 (the vLLM-based line) serving on two RX 7900 XTX; Qwen3.8-27B AWQ-INT4 at TP2 with MTP, 23.8-28.9
  t/s single stream by its README. Read 2026-10-09: it is vLLM plus Python string patches (`build/patches/*.py`)
  that route GEMMs to AITER's Triton kernels (`aiter.ops.triton.gemm_a8w8`) and attention to Triton. It holds no
  gfx1100 version of libr4d's HIP kernels, so it is not a source for the excluded units. What is relevant: its
  all-reduce extensions (`build/patches/radiance_ar_ext.hip`, `radiance_ar_quant_ext.hip`), a router GEMM
  (`router_gemm.hip`), and `docs/fast-reduce-mtp-deadlock.md` on a collective deadlock under graph capture. It needs
  IOMMU off or an ACS override for P2P. GitHub reports no licence for it; the owner stated on 2026-10-09 that the
  repository is their own earlier experiment and may be copied from. The pieces to take are the ones for the
  two-card stage: the all-reduce extensions, the router GEMM and the deadlock note.
- **Both layouts in one file.** llama.cpp `ggml/src/ggml-cuda/mma.cuh`: the same `mma()` for RDNA3
  (`__builtin_amdgcn_wmma_f32_16x16x16_{f16,bf16}_w32`, 16 elements per lane) and RDNA4 (`..._w32_gfx12`, 8 per lane).
- **A libr4d kernel already ported to gfx11 here.** BigCherry patch `1253_nro04_gfx1100_bf16_chunked_gdn`
  (`gated_delta_net_chunked_bf16_gfx11.cu` in the patched tree): libr4d's bf16 GDN chunk scan with gfx11 fragments,
  validated on the XTX cards. The closest model for the GDN and bf16 GEMM units.
- **gfx12 fragment layout, measured.** `github.com/JohnTDI-cpu/rdna4-wmma-guide` (CC BY 4.0): C/D is column = lane % 16,
  row = (lane / 16) * 8 + element. ROCm/ROCm issue 6025 is the documentation gap it fills.
- **gfx11 layout.** On gfx11 lanes 16-31 carry the same A/B data as lanes 0-15, 16 elements per lane (rocWMMA's
  layout traits are the authority; exact file to be pinned).
- **Another RDNA3 WMMA port of a QSA kernel.** `github.com/Niko1221/Strata` PR 856 (gfx12 and gfx11 versions of a QSA
  block-score kernel).
- **No fp8 or fp4 WMMA on gfx11-class cards** is confirmed independently (anthony-chaudhary/fak issue 13542, for
  gfx1151): the fp8 paths have to widen to bf16/f16 or use the int8 path.

File-level pointers from the GPT (Codex) research request req_67189053f04240c6, 2026-10-09. Static evidence from its
reading; not re-verified here except where noted:

- **rocWMMA layout code** (`ROCm/rocWMMA`, branch `develop_deprecated`, `library/include/rocwmma/internal/layout/`):
  `layout.hpp` (`RegisterLayout::Format`, `WMMA_INPUT_GFX11`, `WMMA_ACC_GFX11`), `register_layout_traits_impl.hpp`,
  `register_layout_transforms_impl.hpp`, `transforms/transforms_wmma_impl.hpp` (gfx11 input duplication and
  accumulator padding).
- **LLVM**: `clang/include/clang/Basic/BuiltinsAMDGPU.td` (builtins), `llvm/include/llvm/IR/IntrinsicsAMDGPU.td`
  (intrinsics, including `int_amdgcn_global_load_tr_b128`), and the test
  `llvm/test/CodeGen/AMDGPU/GlobalISel/llvm.amdgcn.wmma_32.ll`: on gfx11 wave32 the f16/bf16 operands are
  `<16 x half>`, iu8 operands `<4 x i32>`, iu4 operands `<2 x i32>`, accumulators `<8 x ...>`. gfx11 has
  `V_WMMA_I32_16X16X16_IU4` with K = 16; gfx12's iu4 is K = 32. This matches the operand widths `r3_compat.h` uses.
- **fp8 through bf16 on gfx11, open source**: `Comfy-Org/comfy-kitchen`, `comfy_kitchen/backends/hip/mma.h`
  (Apache-2.0): gfx11 fp8 to bf16 conversion feeding `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`, with the
  gfx11/gfx12 fragment differences documented. The closest existing code to `r3_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`.
  Widening finite E4M3 values to bf16 is exact; NaN, signed zero and saturation need the numeric decode, which
  `r3_e4m3_to_f32` does.
- **Generic RDNA WMMA GEMM**: `github.com/adelj88/rocm_wmma_gemm`; AMD's guide `gpuopen.com/learn/wmma_on_rdna3/`.
- **Transposed load**: the standard gfx11 replacement is ordinary loads into LDS, an explicit LDS rearrangement, then
  LDS reads into the fragment layout. `r3_global_load_tr_b128_v8i16` does the rearrangement lane to lane instead;
  the LDS form is the faster one to move to.
- Accumulator layouts it states agree with the ones here: gfx11 row = 2 * element + lane / 16, gfx12 row = element +
  8 * (lane / 16), column = lane % 16 on both.

A second request (req_c04e51cd55064790, dev-gpt-agent) asks for a review of `r3_compat.h` itself: the assumed byte
and nibble layouts, the lane id, cheaper lane exchanges, and the extra rounding of the separate accumulator add.

## First kdev run

`flat_add` is faster or equal at every shape (geometric mean 7.2 us against 7.8 us). Up to M = 32 the time is the
launch itself, about 5.5 us. This run proves the loop; the two kernels are deliberately simple.
