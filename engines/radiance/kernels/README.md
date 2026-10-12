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

With radiance's own drafter on (`SPEC=auto`, run `r3-spec1`, the table-lookup build, one XTX): the completion prompt
runs at 131.7 tok/s over 256 tokens and the chat prompt at 171.2 tok/s over 30, same text, server exit 0. The
drafter's kernels work through libr3 unchanged. Both prompts are very predictable (a list of capitals, counting to
ten), so acceptance is far above what mixed prompts give; the R9700's 98-105 tok/s with the drafter was measured on
other prompts, and the two are not comparable until the same prompts are run on both.

**Two RX 7900 XTX in tensor parallel** (`TP=2 GPU=0,1`, runs `r3-tp2-auto`, `r3-tp2-off`, 2026-10-10): serves at
the first attempt. Healthy 6 s after start, 7.51 GiB of weights a rank, the same replies as one card, server exit 0,
and decode **47.0 / 47.2 tok/s** without the drafter (one card: 28.4; the R9700: 37-38). The cards have no
peer-to-peer access to each other ("no peer-to-peer access between cards 0->1, 1->0"), so the collectives go through
pinned host memory; `--p2p off` gives the same 47.1 / 47.3. Not measured yet: prefill, the drafter at two ranks, the
all-reduce share of a step, the lossy wire.

Where the one-card decode time is (run `tools/lab/radiance/libr3-dec11-ablation.sh`, the dec11 kernel with parts
switched off; gate/up GEMM N=34816 K=5120, us a call): everything on 227; no matrix instruction 192; no unpack 197;
no weight loads 186; none of the three 124. So more than half the time is left when the kernel loads no weights,
unpacks nothing and multiplies nothing: it is the loop itself, which waits on memory once every two K steps (a
one-byte exponent load, then table loads that depend on it) while a wave walks its 40 steps. The kernel is bound by
those waits in sequence, not by the instruction, the unpack or the card's memory rate.

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

## Mixed prompts, and the prefill GEMM made native (2026-10-10, runs `p2-*`, `p3-*`, `p4-*`)

`tools/lab/radiance/rad_prompt_bench.py`: eight varied chat prompts (code, explanation, story, arithmetic,
translation, code review, plan, JSON; thinking off, 256 tokens) and the lab corpus cut to about 2,300 and 6,500
tokens, each request uncached, greedy and streamed. MXFP4 Qwen3.8-27B, one run a configuration. Decode is the
geometric mean over the eight prompts (min to max); prefill is at 2.3K / 6.5K tokens.

| Configuration | Decode, no drafter | Decode, drafter | Prefill |
|---|---|---|---|
| R9700, radiance's libr4d | 38.3 | 114.8 (73.6 to 209.4) | 3,104 / 3,214 |
| one XTX, prefill GEMM emulated (`p2`) | 28.2 | 55.0 (30.9 to 97.4), dec11 first form | 32.6 / 31.4 |
| one XTX, prefill GEMM native (`p4`) | 27.9 | 89.0 (53.4 to 158.5) | 1,472 / 1,497 |
| two XTX, prefill GEMM emulated (`p2`) | 46.6 | 91.2 (58.6 to 155.1) | 62.0 / 60.2 |
| two XTX, prefill GEMM native (`p4`) | 46.8 | 141.7 (83.6 to 252.7) | 1,534 / 1,538 |

Prefill on the XTX was about 30 tok/s: the prefill GEMM (`r4d_gemm_mxfp4a8_tiled`) was still going through the
compatibility layer. Its native form (`native/r4d_gemm_mxfp4a8_tiled.hip.rw`, commit f68edac1) brings it to about
1,500 tok/s, 47 times. The selftest passes all three MXFP4 rows with it (decode 81, tiled 10, nt_m64 4). The drafter
figures rose with it because verify steps and the drafter's own passes use the same kernels. Two cards do not
prefill faster than one (1,535 against 1,485); not looked into yet. Prefill is still half the R9700's.

Two cards and the all-reduce (runs `r3-pfprof2`, `r3-wht6`, `r3-wht6spec`). A per-op profile of a prefill-heavy
two-card run puts the fused all-reduce (`ar_rmsnorm_quant_fp8`) at 33% of rank 0's device time, with the MXFP4
GEMMs at about 55%: the two XTX have no peer access, so every all-reduce is exchanged through pinned host memory.
With radiance's lossy wire (`--tp-wire wht6`, a quantised payload for messages of 128 KiB and more) two-card
prefill goes from about 1,535 to 2,124 / 2,121 tok/s, decode without the drafter stays 46.7, with the drafter
131.1 (85.1 to 232.7; 141.7 with the exact wire, one run each, so not a difference to read anything into). The
smoke replies are unchanged; the effect of the lossy wire on output quality has not been measured (`--kld-ref`).

The eight-tiles-a-block form of dec11 (`p3`, commit 6fc8c5cb) was wrong about the cost: decode fell to 15.0 tok/s
(one card) and 24.0 (two), and its ablation with everything off is 112 us for the gate/up GEMM, the same as before,
so the remaining time does not follow the number of workgroups after all. dec11 is not the default; the staged
native form is.

## An 8-bit Qwen3.8-27B: block fp8, and its GEMM made native (2026-10-10, runs `cv-*`, `fp8a-*`, `fp8b-*`)

What radiance serves at 8 bits for this model is block fp8 (E4M3 codes, one bf16 scale a 128x128 block, the form
of Qwen's own FP8 release), not a GGUF Q8_0:

- The Q8_0 GGUF is refused: `no architecture plugin claims 'qwen35' at quantisation 'gguf_ftype_7'`.
- The bf16 GGUF is refused as a source: the qwen35 plugins map the safetensors tensor names
  (`model.language_model.layers.N...`), and a drafter is merged only into a safetensors target.
- From the bf16 safetensors release (`tools/lab/radiance/fetch-27b-bf16.sh`, 52 GB) with radiance's recipe
  `q38-27b-fp8block-df2.recipe`, `tools/lab/radiance/convert-27b-8bit.sh fp8` gives `qwen3.8-27b-fp8-df2.rad`, 29 GB,
  with the DFlash2 drafter merged. It does not fit one 24 GB XTX.
- Int8 codes (`recipes/q38-27b-i8-df2.recipe`, the form whose inner product is the iu8 WMMA gfx11 has) are refused
  by `qwen35_fp8`: `the model did not learn its trunk is int8 before declaring it`. libr4d has the kernels
  (`gemm_i8a8_nt_m16`, `gemm_i8a8_tiled`); the architecture plugin does not build an int8 trunk for this model.

Mixed-prompt bench (`rad_prompt_bench.py`), tok/s:

| | decode | decode, drafter | prefill, 2,300 / 6,500 tokens |
|---|---|---|---|
| R9700, libr4d (`fp8a-r4d`, `fp8a-r4dspec`) | 22.8 | 79.7 | 2,805 / 2,711 |
| two XTX, compatibility layer (`fp8a-tp2`, `fp8a-tp2spec`) | 7.7 | 28.4 | 197 / 209 |
| two XTX, native fp8 GEMM, codes widened to bf16 (`fp8b-tp2`, `fp8b-tp2spec`) | 18.5 | 62.7 | 1,035 / 1,044 |
| two XTX, native fp8 GEMM, codes widened to half (`fp8c-tp2`, `fp8c-tp2spec`) | 44.0 | 125.9 | 1,188 / 1,182 |
| the same with one token row as a dot product and about 1,024 streaming waves (`fp8e-*`) | 48.0 | 124.6 | 1,185 / 1,183 |
| MXFP4 on two XTX, for scale (`p4-*`) | 46.8 | 141.7 | about 1,535 |

The native form is `native/r4d_gemm_fp8a8.hip.rw` (its header says what changes): B staged widened, A widened
where it is read, the accumulator in the gfx11 layout until the epilogue. `r3_selftest` passes `gemm_fp8a8_nt_m16`
(2 cases) and `gemm_fp8a8_tiled` (1) in both forms; `gemm_fp8a8_gated_nt_m16` has no selftest case in radiance and
is covered only by the served text. Quality against MXFP4 is not measured yet.

Widening to half is what moved decode. The bf16 form needs a table for E4M3's eight subnormal codes and ran the
gate/up GEMM at 321 us a call, 139 GB/s of weight (`fp8b-prof`). A half built from the code's own bits
(`(code & 0x7f) << 7 | (code & 0x80) << 8`) is the code's value times 2^-8 for every code, because E4M3 and half
share the subnormal rule, so four codes widen in six bit operations and two v_perm; the f16 WMMA multiplies them
and the result is multiplied by 2^16 once. The same GEMM is then 132 us a call, 337 GB/s (`fp8c-tp2prof`), and the
selftest's host reference still agrees, so the card does not flush half subnormals in the WMMA. Two XTX now decode
the 8-bit model at about twice the R9700's rate; prefill is at 43% of it.

### Weight bandwidth of the decode GEMM (runs `bw-*`, `tools/lab/radiance/libr3-perf.sh`)

An RX 7900 XTX has 960 GB/s of memory bandwidth on paper (384-bit GDDR6 at 20 Gbps), an R9700 640. radiance's
timing mode (`r4d_selftest --perf N K M`) gives each decode GEMM's time over a weight replicated past the caches as
GB/s of the weight stream, M = 1 unless stated:

| kernel, N x K | XTX, libr3 | R9700, libr4d |
|---|---|---|
| `logits_gemm_fp8`, 17408 x 5120 | 792 | 597 |
| `gemm_fp8a16_nt_m1`, 17408 x 5120 | 661 | 593 |
| `gemm_fp8a8_nt_m16`, 17408 x 5120, WMMA tiles (`bw-xtx`) | 523 | 605 |
| the same, 5120 x 8704 | 337 | 595 |
| the same, 8192 x 5120 | 395 | 596 |
| the same, 17408 x 5120 at M = 8 | 637 | 598 |
| one token row as a dot product (`bw-xtx2`): 17408 x 5120 / 5120 x 8704 / 8192 x 5120 | 684 / 314 / 419 | |
| and about 1,024 streaming waves (`bw-xtx3`): the same three | 681 / 664 / 575 | |
| the same: 8704 x 5120 / 2560 x 8704 | 604 / 527 | |

Two separate limits, both design and neither the card:

- **One token row is not a matrix product.** A 16x16 WMMA tile does one column of useful work in sixteen at M = 1.
  As a dot product (a lane multiplies the 32 weight bytes it owns in each fragment-order tile by the activation
  bytes at the same K) the largest shape goes from 523 to 684 GB/s. Served decode did not move with this alone
  (43.4 against 44.0 tok/s, `fp8d-tp2`), because the shapes the two-card split runs were held by the second limit.
- **The XTX wants about a thousand waves streaming at once.** A wave has one 1 KiB request in flight, and libr4d
  stops splitting K at 128 waves, a number fitted on the R9700. N = 5120, K = 8704 runs 320 waves unsplit and reads
  316 GB/s; with the split target at 512 it reads 480, at 1,024 663 and at 2,048 585 (`bw-st*`). N = 17408 is 1,088
  waves unsplit and does not move. libr3 builds with 1,024.

With both, two XTX decode the 8-bit model at 48.0 tok/s (`fp8e-tp2`), which is about 650 GB/s a card averaged over
a whole token and more than the 4-bit container's 46.8. What is left to 960 is the next design step: more than one
request in flight a wave, so that fewer waves are needed and the split's reduce pass goes away. With the drafter
the steps carry several token rows and take the WMMA path, which the new split target does not help (124.6 against
125.9 tok/s; 5120 x 8704 at M = 8 reads 469 GB/s).

## KL divergence: 4-bit against 8-bit, and libr3 against libr4d (2026-10-10, runs `kld2-*`, `kld3-*`)

radiance's KL mode (`rad-kld.sh`) over 24 pieces of the lab's KLD text, 42,761 scored positions, 512-token steps.
No bf16 container of the 27B fits the cards, so the references are quantised models on the R9700 with libr4d.

| reference | scored | KLD mean | p99 | top-1 | PPL against reference |
|---|---|---|---|---|---|
| fp8, R9700 | MXFP4, R9700 (`kld2-mxfp4-r4d`) | 0.0509 | 0.354 | 88.7% | x1.0246 |
| fp8, R9700 | MXFP4, two XTX (`kld2-mxfp4-xtx`) | 0.0508 | 0.361 | 88.7% | x1.0238 |
| fp8, R9700 | fp8, two XTX (`kld2-fp8-xtx`) | 0.0044 | 0.029 | 96.4% | x1.0009 |
| MXFP4, R9700 | MXFP4, R9700 again (`kld3-r4d-again`) | 0 | 0 | 100% | x1.0000 |
| MXFP4, R9700 | MXFP4, one XTX (`kld3-xtx1`) | 0.0037 | 0.027 | 96.8% | x1.0006 |
| MXFP4, R9700 | MXFP4, two XTX (`kld3-xtx2`) | 0.0037 | 0.028 | 96.8% | x0.9993 |

- The 4-bit container is 0.051 from the 8-bit one: a different most likely token at 11% of positions and 2.4%
  more perplexity. That is what the 8-bit model buys, at 44.0 against 46.8 tok/s decode on two XTX.
- radiance repeats exactly on one library and placement, so there is no noise floor to subtract.
- libr3 on gfx11 is 0.0037 from libr4d on gfx12 with the same weights, on one card as on two: it is the kernel
  library, not the two-card split. Perplexity does not move (x1.0006, x0.9993), and per kernel the two libraries
  are equally far from the host reference: `kbench_accuracy.py` over the 537 device cases of `r4d-base2` and
  `libr3-b4` gives the same worst error for every kernel and means that differ only in the GEMMs and attention
  (for example `gemm_fp8a8_nt_m16` 3.0e-06 against 4.1e-06), which are the kernels whose sums the two cards' WMMA
  instructions order differently. So the difference reads as rounding carried through 64 layers of 8-bit
  activation quantisation, not as a less accurate kernel. That fixture predates the native forms and has no MXFP4
  rows; those are checked by `r3_selftest` against the host reference, not against libr4d bit for bit.

## RR08: lane-owned MXFP4 prefill on gfx1100 (measured 2026-10-12: slower, not adopted)

R3_BT=1 selects libr3's r3_gemm_mxfp4a8_bt_gfx11 implementation inside the existing
gemm_mxfp4a8_tiled registry row. It is default-off; R3_BT=0 or unset keeps the established tiled
implementation. The higher-priority decode rows and their entry points are unchanged. R3_BT_B=4, 6 or 8 selects
how many 16-row activation tiles reuse one weight fragment (default 8); R3_BT_NT=0 or 1 controls the
non-temporal weight load (default 1).

The implementation deliberately changes RR08's proposed bf16 widened representation to f16. The block-fp8
measurements above moved the same gfx1100 WMMA path from 321 us / 139 GB/s with bf16 widening to 132 us /
337 GB/s with bit-built f16. After MXFP4's E2M1 code and E8M0 exponent are folded to the E4M3 byte libr4d
defines, the same widening applies: both operands are exactly 2^-8 of the E4M3 value and the epilogue applies
2^16 once. The RR06 host oracle is the correctness gate on this change.

A block is one 32-thread wave. Lane l owns weight row n0 + (l & 15), reads its two fragment-order dwords for a
16-K step, expands them in registers and reuses the f16 fragment across B activation tiles. Incoming E4M3
activations are widened once into row-major f16 in the op's scratch; a lane's 16 values are then one 32-byte
load. Accumulators remain in gfx1100's interleaved layout through the epilogue. No Radiance source or ABI change
is required: libr3's existing native rewrite makes r4d_rad_gemm_mxfp4a8_tiled_scratch allocate two bytes per
rounded activation element, up to about 17.8 MiB at M=512, K=17408, and the alternate launcher consumes that
same scratch.

The decode result that one token row is not a matrix product does not apply at M=64..512: the WMMA activation
columns are useful. The other gfx1100 result does apply: the card wants roughly a thousand streaming waves.
The wide prefill grids naturally provide them, so this form does not add split-K. Its expected first limiter is
BT accumulator VGPR pressure, then repeated activation reads across N tiles. If activation traffic is the
limiter, the second form is RR08's eight-wave block with one LDS-staged activation group.

Validation uses one built libr3 for both arms. R3_BT_ORACLE=1 expands the RR06 host-reference matrix to all five
Qwen3.8-27B prefill projections at M=64, 128 and 512 on both activation-scale grids. The ordinary
gemm_mxfp4a8_tiled r3_selftest case checks the selected arm quickly. libr3-perf.sh --perfmxfp4 rotates the
codes, E8M0 plane and Wref beyond cache and includes activation widening in the measured row time. Promotion
requires all oracle cases to pass, decode to remain unchanged, and one-XTX long-prompt prefill to reach at least
2,500 tok/s; the established native tiled path is about 1,500 and libr4d on the R9700 is about 3,200.

| B | NT | oracle worst rel_l2 | 34816x5120 M512 GB/s | 5120x17408 M512 GB/s | one-XTX 6.5K tok/s | two-XTX wht6 6.5K tok/s |
|---:|---:|---:|---:|---:|---:|---:|
### Result (2026-10-12, runs `rr08a-*`, `tools/lab/radiance/rr08-cycle.sh`; one RX 7900 XTX unless stated)

Builds (62 units for gfx1100) and passes `r3_selftest` `gemm_mxfp4a8_tiled`, 10 of 10 cases, with `R3_BT=0` and with
`R3_BT=1`. Row timing (`--perfmxfp4`, us a call; N x K, M = token rows), established tiled form against the
lane-owned form with `R3_BT_B=8`:

| N x K | M = 64 | M = 128 | M = 512 |
|---|---|---|---|
| 34816 x 5120 | 915 / 781 | 930 / 964 | 1,807 / 3,754 |
| 5120 x 17408 | 545 / 887 | 548 / 902 | 1,043 / 2,632 |
| 16384 x 5120 | 457 / 383 | 476 / 451 | 854 / 1,863 |
| 5120 x 6144 | 187 / 306 | 194 / 318 | 366 / 1,028 |
| 12288 x 5120 | 316 / 325 | 302 / 349 | 620 / 1,442 |

`R3_BT_B=4` and `6` are slower than 8 at every shape (for example 16384 x 5120 at M = 512: 2,109 and 2,273 us).

Serve bench (`rad_prompt_bench.py`, MXFP4 Qwen3.8-27B, prefill at about 2,570 / 7,175 tokens, tok/s):

| | tiled (`R3_BT=0`) | lane-owned (`R3_BT=1`) |
|---|---|---|
| one XTX, prefill | 1,544 / 1,504 | 728 / 717 |
| two XTX, `--tp-wire wht6`, prefill | 2,013 / 2,170 | 1,259 / 1,271 |
| decode, 8 short prompts (one / two XTX) | 28.1 / 47.0 | 28.2 / 46.6 |

- The lane-owned form is 2.1 to 2.8 times slower at M = 512, the size prefill runs, and prefill halves. It is ahead
  only at M = 64 on the two shapes with N much larger than K. Decode is unchanged, as intended.
- The tiled form's time grows 1.9 times from M = 64 to 512 (eight times the rows); the lane-owned form's grows 4.8
  times. The WMMA tiles amortise the weight stream over 16 token rows at once and the direct form does not: its
  activation loads repeat for every N tile. That is the reuse assumption failing, which is this item's stop
  condition, so the direct one-wave form is not tuned further. `R3_BT` stays off.
- What the table says about the tiled form itself: at M = 512 it moves 45 to 54 GB/s of weight, a tenth of what the
  card streams in decode, so prefill is compute-bound in the WMMA product and its activation handling, not
  memory-bound. A form that reaches 2,500 tok/s on one card has to roughly halve the M = 512 column.
- Next form, as planned: eight waves a block sharing one LDS-staged activation tile, lane-owned f16 weights and
  accumulators kept.

## RR08 second form: eight-wave LDS-shared activation tile (R3_BT=2; awaiting hardware)

The M=512 tiled measurements are already close to gfx1100's matrix roofline. A 16x16x16 WMMA is 8,192
floating-point operations, so the exact instruction count is (M/16)*(N/16)*(K/16). Against the RX 7900 XTX's
123-TFLOP/s FP16/BF16 matrix peak, the arithmetic-only floor is:

| N x K | M512 WMMAs | 123-TF floor | measured tiled | floor / measured |
|---|---:|---:|---:|---:|
| 34816 x 5120 | 22,282,240 | 1,484 us | 1,807 us | 82% |
| 5120 x 17408 | 11,141,120 | 742 us | 1,043 us | 71% |
| 16384 x 5120 | 10,485,760 | 698 us | 854 us | 82% |
| 5120 x 6144 | 3,932,160 | 262 us | 366 us | 72% |
| 12288 x 5120 | 7,864,320 | 524 us | 620 us | 84% |

Therefore no re-tiling that still issues the same dense f16/bf16 WMMAs can halve M=512 time. Removing every
non-WMMA cycle gives only about 1.18-1.41x. The new form is an ablation-driven attempt to recover that remaining
16-29%, not a 2x claim.

At M=512 the existing TN=2/LBK=128 block is 256 M x 64 N, eight waves, 512 WMMAs per K slab. In libr3 it stages
64x128 weights as bf16: about 4 KiB packed codes plus 0.25 KiB E8M0 are read globally, about 17 KiB including
padding is written to LDS, and the four M-waves reuse each N-wave's weight fragments for about 64 KiB of LDS reads.
Its activation fragments are direct from the widened scratch: about 128 KiB of global/cache loads per block-slab
because the two N waves load the same 256x128 activation slab independently. The block carries eight v8f
accumulators per wave (64 f32 VGPR values) plus up to four activation and two weight fragments; LDS is only about
18 KiB, so register pressure is the likely occupancy limiter.

R3_BT=2 uses an equal-output-area 128 M x 128 N block at B=8. Eight N-waves share one tile-major 128x128 f16
activation slab (32 KiB LDS): about 32 KiB activation global loads, 4x less than tiled; weights stay lane-owned,
so there are no weight-LDS stores/reads and no final native-to-gfx12 accumulator shuffle. The trade is 8 KiB of
packed weight per slab and twice as many weight values expanded for the same output area. It keeps only one live
activation and one live weight fragment beside the same eight accumulators, so the compiler resource report should
show lower VGPR pressure; actual resident waves need the lab build to confirm.

The MXFP4 E8M0 exponent can be applied after an integer dot only for the weight side. It does not make the whole
product an exact IU8/IU4 dot: E2M1 magnitudes need signed values up to 12 after the natural x2 integer mapping, and
each E4M3 activation has its own element exponent in addition to the row/block scale. Factoring one exponent after
an integer product therefore cannot reproduce the current product. An exact integer route needs decomposition into
multiple products and loses the matrix-rate advantage; a lossy activation requantisation would be a different
numeric contract.

R3_BT_ABL is timing-only and off by default. Bits: 1 replaces weight expansion with constants; 2 replaces activation
widening with constants while retaining the pre-pass writes; 4 skips WMMA; 8 skips the epilogue/store. Non-zero
masks intentionally produce garbage.

Run the ablations first, then correctness/perf/serve:

    BT=2 STEPS="build self abl" ABL=1,2,4,8 bash tools/lab/radiance/rr08-cycle.sh rr08b-abl
    BT=2 ORACLE=1 STEPS="self" bash tools/lab/radiance/rr08-cycle.sh rr08b-oracle
    BT=2 BT_B=8 STEPS="perf" bash tools/lab/radiance/rr08-cycle.sh rr08b-perf
    BT=2 BT_B=8 STEPS="serve1 serve2" bash tools/lab/radiance/rr08-cycle.sh rr08b-serve

Row timing, us/call, tiled / R3_BT=2:

| N x K | M=64 | M=128 | M=512 |
|---|---:|---:|---:|
| 34816 x 5120 |  |  |  |
| 5120 x 17408 |  |  |  |
| 16384 x 5120 |  |  |  |
| 5120 x 6144 |  |  |  |
| 12288 x 5120 |  |  |  |

Ablation at M=512, full / mask time in us:

| mask | meaning | tiled | R3_BT=2 |
|---:|---|---:|---:|
| 1 | weight expansion -> constants |  |  |
| 2 | activation widen -> constants |  |  |
| 4 | skip WMMA |  |  |
| 8 | skip epilogue/store |  |  |

Serve:

| | tiled | R3_BT=2 |
|---|---:|---:|
| one XTX, 2.5K / 6.5K prefill tok/s |  |  |
| two XTX wht6, 2.5K / 6.5K prefill tok/s |  |  |
| decode one / two XTX |  |  |

Stop tuning this form if the M=512 mask-4 runs show the non-WMMA remainder at roughly 30% or less and the full
R3_BT=2 arm does not materially reduce the tiled column: the fixed WMMA count then prevents the required ~2x.
The next exact form should stay close to the existing weight-staged tiled kernel and change tile aspect/live
fragments (for example 128 M x 128 N, fewer live A/W fragments) to push toward the 123-TF roofline. Reaching ~2x
requires a higher-throughput arithmetic representation or a changed numeric contract, not another LDS placement.

## The lossy two-card wire, measured (2026-10-12, runs `kld4-*`, `tools/lab/radiance/kld-tp-wire.sh`)

`--tp-wire wht6` took two-card MXFP4 prefill from about 1,535 to 2,124 tok/s (above) with no quality number. KL mode
over the same 42,761 positions, both rows against MXFP4 on the R9700 with libr4d:

| two XTX, libr3 | KLD mean | median | p99 | max | top-1 | PPL against reference |
|---|---|---|---|---|---|---|
| exact wire (`kld4-xtx2-exact`) | 0.003665 | 0.001643 | 0.0275 | 1.78 | 96.79% | x0.9993 |
| `wht6` (`kld4-xtx2-wht6`) | 0.004422 | 0.001980 | 0.0314 | 4.67 | 96.38% | x1.0003 |

- The exact row repeats `kld3-xtx2` to the last digit, so the difference between the rows is the wire alone:
  +0.0008 mean KLD, 0.4 points of top-1 agreement, perplexity unchanged (x1.0003 against x0.9993).
- For scale, the 4-bit container is 0.051 from the 8-bit one (88.7% top-1): the wire costs about a sixtieth of what
  the 4-bit quantisation already does. The largest single-position error is larger with the wire (4.67 against 1.78).
- So `--tp-wire wht6` is the two-XTX serving default from here: +38% prefill, decode unchanged (46.7 against 46.8
  tok/s; with the drafter 131.1 against 141.7 was one run each and is not established either way).

## Where a prefill's time goes: one XTX, two XTX, the R9700 (2026-10-12, runs `pp1-*`, `prefill-profile.sh`)

MXFP4 Qwen3.8-27B, one 7.2K-token prompt, `--profile-ops` (so absolute speeds are the profiled ones), device ms
summed by op (`profile_ops_summary.py`); for two XTX the figures are one rank's, with `--tp-wire wht6`.

| op | one XTX | two XTX, a rank | R9700 (libr4d) |
|---|---|---|---|
| prefill, tok/s | 1,467 | 2,137 | 3,206 |
| all ops, ms | 9,439 | 6,300 | 3,783 |
| `gemm_nt_q` (MXFP4 GEMMs) | 7,983 (84.6%) | 4,287 (68.1%) | 3,013 (79.6%) |
| `ar_rmsnorm_quant_fp8` (fused all-reduce) | - | 1,095 (17.4%), 145 us a call | - |
| `attn_paged` | 619, 656 us a call | 353, 374 us | 135, 143 us |
| `gdn_chunk_scan` | 300, 417 us | 253, 351 us | 95, 133 us |
| the rest | 537 | 312 | 540 |

- Two cards give 1.46x one. The GEMMs split almost in half (0.537 of one card's time a rank); what does not scale
  is the all-reduce (17% of a rank's time even on the lossy wire) and the delta-net scan (300 -> 253 ms).
- Against the R9700 a rank of two XTX is 2,517 ms behind: 1,274 in the GEMMs, 1,095 in the all-reduce, 218 in
  attention, 158 in the scan. Attention and the scan are 2.6 to 4.6 times slower a call on the XTX than on the R9700
  and are still compatibility-layer forms; together they are about 10% of a rank.
- GEMM throughput: the tiled kernel does a 16384 x 5120 product at M = 512 (85.9 GFLOP) in 854 us, about 100
  TFLOP/s; AMD states 123 TFLOPs peak FP16 matrix (and 123 TOPS int8) for the RX 7900 XTX and 191 / 383 for the
  R9700. The R9700's 3,206 tok/s on a model of about 52 GFLOP a token is about 167 TFLOP/s. Prefill is bounded by
  matrix arithmetic, not memory bandwidth (the tiled form streams 45 to 54 GB/s of weight at M = 512).
- Arithmetic of what is left for two XTX: all-reduce to zero gives about 2,660 tok/s; attention and the scan at the
  R9700's per-call cost about +6%; GEMMs at 115 instead of 100 TFLOP/s about +10%. Together about 3,000 to 3,300:
  level with one R9700 if all three land, on the stated 123 TFLOPs roof.
- That roof is not yet measured here. With the WMMA instructions ablated (`R3_BT_ABL=4`) the same call takes 437 us,
  which would leave only 417 us for 85.9 GFLOP (over 200 TFLOP/s) - but removing the products lets the compiler drop
  the loads that feed them, so this is not a measurement of the instruction rate. A registers-only WMMA loop is.

### The engine's own options on two XTX (2026-10-12, runs `knob1-*`, `tools/lab/radiance/tp2-knobs.sh`)

Serve bench, unprofiled, one run an arm; prefill at about 2.6K / 7.2K tokens, decode over 8 short prompts:

| arm | prefill, tok/s | decode, tok/s |
|---|---|---|
| `--tp-wire wht6`, chunks of 2048 (base) | 2,112 / 2,125 | 47.1 |
| chunks of 4096 | 2,192 / 2,104 | 46.7 |
| chunks of 1024 | 2,090 / 2,054 | 46.5 |
| `--tp-wire-min-kb 16` | 2,155 / 2,154 | 46.4 |
| fp8 KV cache | 2,150 / 2,147 | 45.5 |
| exact wire | 1,546 / 1,561 | 46.2 |

Nothing but the wire moves prefill by more than a single run's spread (about 2%): chunk size does not change the
all-reduce's cost a token, the lossy wire is already used for nearly all prefill bytes at the 128 KiB threshold, and
the KV dtype is not where the time is. What is left is in the kernels and in how the all-reduce is done.

### RR08, second form (`R3_BT=2`, eight waves sharing an LDS activation tile; runs `rr08b-*`)

Passes the selftest (10 of 10). Slower than the tiled form at M = 512 on every shape (16384 x 5120: 1,553 against
854 to 895 us; 5120 x 6144: 640 against 366; 12288 x 5120: 1,107 against 620) and in serving (one XTX 984 / 965
tok/s against 1,399 / 1,475). Not adopted; `R3_BT` stays 0. Ablation of the tiled form at 16384 x 5120, M = 512
(895 us): weight expansion as constants 785, activation widen as constants 889, no epilogue 881, no WMMA 437.

## The WMMA arithmetic roof of one XTX, measured (2026-10-12, run `rr09c`, `tools/lab/radiance/wmma-roof.sh`)

`r3_selftest --perfwmma 8192` on HIP device 0 (48 CUs): the WMMA instruction alone, operands held in registers, swept
over waves a block, blocks a CU and independent accumulators; best of each element type:

| operands | best | geometry |
|---|---|---|
| f16 | 142.0 TFLOP/s | 2 waves a block, 4 blocks a CU, 8 accumulators |
| bf16 | 142.4 TFLOP/s | 1 wave a block, 8 blocks a CU, 8 accumulators |
| iu8 | 144.1 TOPS | 2 waves a block, 8 blocks a CU, 8 accumulators |
| iu4 | 285.8 TOPS | 8 waves a block, 4 blocks a CU, 8 accumulators |

- The roof is above AMD's stated 123 TFLOPs. The tiled MXFP4 prefill GEMM at about 100 TFLOP/s is at 70% of it, so
  the most a perfect GEMM could return is 1.4x on the 68% of a two-card prefill that is GEMM.
- iu8 gives no more arithmetic than f16 on this card (144 against 142); iu4 gives twice, but needs both operands in
  4 bits, and an int8 activation split into two nibbles costs two multiplies, which lands back on the iu8 figure.
- The bench's `lds` and `global` rows (4.8 and 1.8 TFLOP/s) reload both operands for every multiply; they are not a
  bound on a GEMM, which loads an operand tile once for many multiplies. They are not used above.
- The all-reduce phase trace (`ar-phase-profile.sh`, runs `rr10arb-*`) gave nothing: the engine aborts under rocprofv3
  in `r4d_ar_ipc_alloc` -> `hipMemset` -> `rad_aql_blit_fill` while declaring the graph, and the script still printed
  its DONE line over empty tables. The phase split has to come from timers inside the plugin, not from a tracer.

## Native prefill attention and delta-net scan, measured (2026-10-12, run `rr10d`, `tools/lab/radiance/rr10-native-prefill.sh`)

27B MXFP4, two XTX, `--tp-wire wht6`, one 7.2K prefill a server start; switches `R3_ATTN_NATIVE` and `R3_GDN_NATIVE`
(both default off: see the correction below). Selftests pass in every arm (`attn_prefill_h256_gqa6_fp8kv`, `_bf16kv`,
`gdn_chunk_scan_k128_v128_c64_bf16`).

| arm | prefill tok/s | `attn_paged` us a call | `gdn_chunk_scan` us a call |
|---|---|---|---|
| compatibility forms | 2,187.8 | 369 | 349 |
| native attention | 2,306.6 | 204 | 350 |
| native scan | 2,283.4 | 408 | 127 |
| both | 2,330.5 | 234 | 127 |
| R9700, libr4d (for scale) | 3,104 to 3,214 | 143 | 133 |

- The scan is now at the R9700's cost a call; attention is at 1.4 to 1.6 times it (from 2.6 times).
- Repeats (`rr10e`, `rr10f`, same script): compatibility 2,187.8 / 2,134.9 / 2,156.7; native attention 2,306.6 /
  2,246.5 / 2,262.3; native scan 2,283.4 / 2,221.0 / 2,230.1; both 2,330.5 / 2,335.2 / 2,367.0. Both against
  compatibility is complete separation over three starts an arm, mean 2,344 against 2,160 (+8.5%); selftests pass in
  every round. Both native forms are now the default.
- CORRECTION (runs `kq0`, `kq1`, `kld-tp-wire.sh` against MXFP4 on the R9700): with both native forms on, two-XTX
  output is wrong - KLD mean 6.50, top-1 agreement 7.3%, PPL 4,908 against 8.56 - on the exact wire and on wht6
  alike. With both off the same check gives KLD 0.0037 / 96.8% (exact) and 0.0044 / 96.4% (wht6), as before. The
  selftests compared each kernel with the compatibility form on their own rows and passed; they do not cover what
  the served model asks of these kernels. The +8.5% above is the speed of a wrong answer. Both are off again until
  each is fixed and passes this check.
- One switch at a time, every switch set explicitly (runs `kqa2`, `kqg2`, `kqw2`; the KL script does not rebuild
  libr3, so a default changed in source is not what an unrebuilt plugin does):
  - `R3_ATTN_NATIVE=1` alone: KLD 0.0035 / 96.8% exact, 0.0044 / 96.6% wht6 - as good as the compatibility form.
    It is the default again (prefill 2,247 to 2,307 against 2,135 to 2,188 in the repeats above, +5%).
  - `R3_GDN_NATIVE=1` alone: KLD 6.50 / 7.3% - this is the broken one. Off.
  - `R3_AR_WIRE4=1` with wht6: identical to plain wht6 to every digit (0.004422 / 96.382%), and the 4-bit run's
    prefill was no faster: the 4-bit path is not reached by the served model. Not adopted.
  - After GPT's fixes (9448b430, 9ced23c0, and a6850cb9 for a gfx12-only store wait that broke the gfx1100 build):
    `R3_GDN_NATIVE=1` still wrong (KLD 5.52, top-1 11.5%, PPL 1,862; run `kqg4`), off.
    `R3_AR_WIRE4=1` now engages (prints `R3_AR_WIRE4 active`): prefill 2,376-2,387 against 2,251-2,256 tok/s,
    alternating, three starts an arm (+5.6%, complete separation); quality with wht6 KLD 0.0126 / 94.5% top-1 /
    PPL 8.605 against 0.0044 / 96.6% / 8.553 for wht6 alone (run `kqw4`). Three times the wire's divergence for
    5.6%: left opt-in, the owner's call.
  - `R3_GDN_NATIVE=1` after 0b8b7b06 (accumulation matched, `R3_GDN_COMPARE` in-plugin comparison): the comparison
    run printed "no mismatch" but no comparison line at all, and the KL check is unchanged (KLD 5.52, top-1 11.5%,
    run `kqg5`). Shelved as evaluated, default off: the scan is about 4% of a 27B prefill, so a correct native form
    is worth 2-3%. The fault is somewhere the comparison does not reach.
- The 4-token decode figures in this run (a 71.3 maximum, a 64.0 after the prefill) are too short to mean anything.

## 27B on two XTX after the native attention and scan: what is left (2026-10-12, run `pp3`)

`prefill-profile.sh` (`CONFIGS=xtx2`), 7.2K prefill, a rank: `gemm_nt_q` 4,132 ms (70.1%), `ar_rmsnorm_quant_fp8`
1,177 ms (20.0%, 6,771 calls, 174 us), `attn_paged` 197 ms (3.3%), `gdn_chunk_scan` 92 ms (1.6%), everything else
5%. Profiled times are inflated by the restored synchronisations; the shares are what to read.

- Unprofiled the pair does 2,344 tok/s, the R9700 3,104 to 3,214: the pair has to shed about 27% of its prefill time.
- The cross-card sum is a fifth of it. A reduce of a 2048-token chunk carries 2048 x 5120 values at 6 bits, 7.9 MiB,
  off one card into pinned host memory and then onto the other; two reduces a layer over 64 layers and four chunks is
  about 500 large reduces, so roughly 2.3 ms each - what two one-after-the-other crossings of an x8 link cost for
  7.9 MiB. That reading (the sum is link time for the wire's bytes, paid twice in series) is an estimate from the
  counts, not a measurement; the plugin timers are to confirm it. If it holds, overlapping the two crossings
  (the receiving card starts on the first slices while the sender is still copying out) is worth about half of it.
- With the sum at half and the GEMM at 115 TFLOP/s instead of 100, the pair would be at about 2,900 tok/s; matching
  the R9700 needs the sum near a quarter of today's and the GEMM near 125 (the roof is 142).

## The cross-card sum, timed: it is the link, by the byte (2026-10-12, run `art3`, `tools/lab/radiance/ar-timing.sh`)

`R3_AR_TIMING=1` times each fused all-reduce call between two stream synchronisations (the engine's HIP layer has no
`hipEventRecord` on its streams, and aborts under rocprofv3), so the timed run is synchronous and slower; reported
every 512 calls a row. 27B MXFP4, two XTX, `--p2p off`, 7.2K prefill. Without P2P the kernel itself writes its
payload into mapped pinned host memory and reads the peer's from it; there are no separate copies to time.

| wire | rows | payload a call | p50 | p90 | calls | total a rank |
|---|---|---|---|---|---|---|
| wht6 | 512 and up (the 2048-token chunks) | 7.17 MB | 2.0 to 2.15 ms | 2.6 to 2.8 ms | 508 | 1.00 to 1.02 s |
| exact | 512 and up | 18.35 MB | 5.0 to 5.1 ms | 5.5 to 5.6 ms | 512 | 2.29 to 2.31 s |
| wht6 | 9 to 511 | 0.19 to 0.28 MB | 104 to 151 us | 167 to 251 us | 1,143 | 0.16 s |
| exact | 1 to 8 (decode) | 10 to 36 KB | 43 to 74 us | 54 to 160 us | | |

- The large reduces are the whole cost (1.0 s of the 1.18 s the per-op profile gives the op), and their time follows
  the bytes: 3.4 to 3.6 MB a millisecond on either wire. A call moves its payload out and the peer's in, so the link
  is carrying about 7 GB/s; the estimate above (about 2.3 ms, two crossings one after the other) holds.
- So the sum gets cheaper only by fewer bytes or by the two crossings overlapping. Out and in are opposite
  directions of the card's own link: a kernel that writes its payload in slices while reading the peer's slices as
  they land would take about one crossing instead of two (about 1 ms), which is half of the 20% the sum costs a
  prefill. A 4-bit wire would take a third off the bytes, at a quality cost to be measured in KL mode.
- Decode-size reduces are 43 to 74 us; nothing to gain there.
- `R3_AR_PIPE=1` (eight slices with per-slice ready flags in one kernel, run `arp1`, `art4`): selftest byte-identical
  at M=64, but slower. The large wht6 reduce goes from 2.0-2.15 ms to 2.62-2.74 ms a call, and 27B prefill from
  2,323-2,358 tok/s to 2,170-2,224 over three starts an arm (complete separation). Left off. Slicing inside one
  kernel does not make the out and in crossings run together: the waves that read wait on flags the same kernel's
  writers have not reached, and the flag polling over the link adds traffic of its own.
- `R3_AR_WIRE4=1` is the next opt-in candidate (not measured yet): only the pinned-host two-rank fused
  `ar_rmsnorm_quant_fp8` path with `wire=wht6` and `M >= 512` changes; smaller prefill calls and decode retain
  wht6 exactly. It keeps wht6's 64-value orthonormal Walsh-Hadamard block and one BF16 scale per block, but uses
  `scale = bf16(amax / 7)`, RNE `rintf(value / scale)`, clamp [-8, 7], then stores code+8 in four bits. A
  512-element chunk is 256 code bytes plus 16 BF16 scale bytes (17/32 byte per element), against wht6's 384+16
  (25/32), so the measured 7.17 MB large payload should become about 4.87 MB.
  The shared exact scratch keeps its full `slot_bytes` stride/capacity; wire4 writes codes from the slot base and
  moves the peer BF16 scale plane from `slot_bytes/2` to `slot_bytes/4`. `locpack` likewise keeps codes at
  offset zero and moves its local f32 scale plane from `slot_bytes/2` to `slot_bytes/4`. The lab gate is
  `tools/lab/radiance/ar-wire4.sh`: random and heavy-tailed M=2048 error against exact (with wht6 beside it),
  three 27B MXFP4 starts per arm, then KLD/top-1 against an exact reference.

## Qwen3.8-Flash-Next on two XTX, first runs (2026-10-12, runs `fn2`, `fnp1`, `tools/lab/radiance/flashnext-tp2.sh`)

radiance's shipped container `qwen3.8-next-flash-fp8-iq4r-moe.rad` (113.6 GiB: 4-bit codebook experts `w4nl64a8h`,
int8 trunk, MTP head, vision), radiance 1.3.0 (89cee7c), libr3, `--tp 2 --tp-wire wht6 --placement expert_tiered
--host-pool-mib 32768 --gpu-headroom-mib 256 --kv-cache-dtype fp8`, context 32768, no speculation. (The container's
card also passes `--expert-vs-cache-ratio 0.82`; 1.3.0 does not have that option.)

- It loads and serves: 2,137 declared ops, none without a libr3 kernel; smoke replies correct; clean exit.
- Placement a card: static weights 3.44 GiB, experts 19.00 GiB, KV 271 MiB; resident expert units a layer mean
  118.7 of 512 (max 444); the n-gram table (47.68 GiB) is read from disk because it fits neither a card nor the
  host pool beside the experts.
- Decode 28.3 tok/s over 8 prompts (27.0 to 29.7). Prefill 2,575 tokens in 14.5 s and 7,176 in 29.2 s (about 178 and
  245 tok/s); the bench client timed out, the figures are the server's ttft. During prefill the engine reports
  synchronisations waiting over 5 s on `moe_gemm_q_kernel`.
- For scale: radiance's own figures on two R9700 are 15.3 ms a decode step and about 6,500 tok/s at 26K
  (`docs/MOE-W4.md`); BigCherry llama.cpp on this host does about 1,465 tok/s prefill on the IQ4_XS GGUF.

Per-op profile of the same bench (`fnp1`, rank 0, 95.8 s of device and host-site time, decode-heavy):

| op | share | calls | us a call | |
|---|---|---|---|---|
| `moe_gemm_q` | 40.9% (46.2% on rank 1) | 190,560 | 205 (232) | routed-expert GEMM, compatibility layer |
| `embed_lookup_q` | 13.8% | 1,985 | 6,644, host site | the n-gram table read from disk every step |
| `gemm_nt_q` | 11.4% | 428,760 | 25 | int8 trunk GEMMs |
| `hc_read` | 11.3% | 192,545 | 56 | hyper-connection read |
| `ar_gather_hc_write` | 7.5% | 95,280 | 75 | fused all-reduce + hyper-connection write |
| `attn_paged_gate_quant` | 2.9% | 23,688 | 116 | |
| `router_gemm_topk_scatter` | 2.3% | 95,088 | 23 | |
| `qsa_score` | 1.8% | 23,820 | 72 | |

The n-gram table does not fit in this host's memory beside the experts (runs `fn3`, `fn4`). `--host-pool-mib` is a
pool for each rank: with 24576 the engine refuses `--ngram-placement ram` (85.66 GiB available, 47.68 GiB table plus
49.19 GiB of pools); with 10240 the table fits but the experts do not (each rank needs 19.13 GiB of pool beyond the
19.00 GiB on its card). The sum that would fit, 2 x 18.5 GiB + 47.68 GiB, leaves the 91 GB host under 1 GiB free and
needs another 0.6 GiB of experts on each card, so it is not a configuration to serve from. The disk lookup therefore
stays, at 6.6 ms a decode step (about a fifth of the step); it is a small part of a prefill (1,985 calls in the run).
Loading radiance's host library changes nothing (`KERNELS=libr3,libavx,libref`, runs `fn5` profiled, `fn6` not): the
lookup is 6,566 us a call averaged either way (about 4.2 to 4.4 ms on decode steps), decode 28.1 tok/s against 28.3.
The cost is the design's own: libavx keeps 2^19 rows and reads a miss from the container with O_DIRECT, and about half
of a decode step's rows are new. A random 4 KiB direct read on this drive is about 200 us at one thread
(`tools/lab/radiance/direct-read-latency.py`), the same on `/mnt/vault` (NTFS through FUSE) and on ext4 `/mnt/data`,
so the filesystem is not the cause. Removing it means holding the table somewhere other than the disk: on the cards
outside the rank group (the R9700 takes about 30 of the 47.68 GiB), which is asked of the plugin as its own item.

Order of work this gives: a native gfx11 `moe_gemm_q` first (two fifths of everything); then `hc_read`
and `ar_gather_hc_write`.

## Flash-Next on two XTX: the experts that do not fit are the cost (2026-10-12, runs `fnhc1`, `fn7`, `mps2`)

Each rank's experts are 38 GiB and a 24 GB card keeps 19.00 GiB of them; the rest sit in the pinned host pool and
are read over the link by the kernel, or swapped in by the heat engine.

- The heat engine is busy for the whole bench (`fnhc1-a0`, a card): 108,048 dispatches, 21,375 promotions, 20,934
  demotions, 64% of promotions of a unit already demoted once, 25.25 GiB moved to the card and 22.87 GiB back;
  resident units a layer mean 118.8 of 512 (min 0, max 458).
- Stopping it is worse (`--deterministic`, run `fn7`): decode 19.9 tok/s against 29.2, prefill 178.6 / 229.1 tok/s
  against 211.7 / 411.4; `moe_gemm_q` 330 us a call against 204, `ar_gather_hc_write` 353 against 126. So the
  swapping earns its traffic, and what it cannot make resident is what costs.
- The expert GEMM alone against the share of experts read from host memory (`moe-pool-sweep.sh`, one XTX,
  `r3_selftest --perfmoe`, the `down` projection, best form of each row):

  | experts in host memory | M=1 | M=2048 |
  |---|---|---|
  | 0% | 78 us | 5.13 ms |
  | 50% | 78 us (no routed row landed on a pooled expert) | 13.4 ms |
  | 75% | 79 us (same) | 24.7 ms |
  | 100% | 466 us | 47.1 ms |

  A prefill chunk's expert GEMM is 9 times dearer with every expert over the link than with all resident, and about
  5 times at the three quarters this model has off a 24 GB card; a decode call that lands on a pooled expert is 6
  times dearer. The link delivers 3 to 5 GB/s of weight to the kernel.
- The form that is fastest resident (2 and 4) is the slowest pooled, and form 1 the reverse, at M=2048.
- Even resident, the w4 expert GEMM does 6.5 TFLOP/s useful at M=2048 (about 40 rows an expert) against a 142 TFLOP/s
  arithmetic roof, so a native form has room as well.
- `R3_HC_NATIVE=1` (native E4M3 decode for `hc_read`): 58.2 -> 50.7 us a call, decode 29.2 -> 29.7 tok/s, inside the
  start-to-start spread; left off.
- `R3_MOE_NATIVE=1` (full 16-code rows and the accumulator kept in gfx11 form through a scale group, run `fnmoe1`):
  `moe_gemm_q` 187.5 -> 175.5 us a call, decode 29.8 -> 30.9 tok/s, prefill 160.9 / 330.4 -> 205.3 / 373.4 tok/s on
  one start an arm - the same size as the spread between starts of one configuration (330 to 411 at 7.2K). A 6% cut
  of the call is what an inner-loop change can give while most of the call is the link; left off until repeated.

What follows: the experts off the two cards should be computed where they can be resident (the R9700, 30 GiB, then
the 6900 XT), with only activations and results crossing host memory - about 5 MiB each way for a 2048-token chunk
against several hundred MiB of weights a layer - while the hot set stays on the XTX pair as now.

## Flash-Next: the n-gram table on the R9700 and in host memory (2026-10-12, run `fnng7`, `tools/lab/radiance/fn-ngram-dev.sh`)

`R3_NGRAM_DEV=2` (default off): libr3 offers the host kernel for `embed_lookup_q` (the engine keeps the table's
placement `disk`), and gathers the rows on a card outside the rank group through `libr3_ngram_helper.so` (built for
gfx1100, gfx1201 and gfx1030, kept in `radiance_home/helpers` because the engine loads every library in `kernels` as a
plugin). The rows the device does not hold are copied into host memory and gathered from the copy by libavx's kernel
as memory; nothing is read from the container after the load. `R3_NGRAM_RAM=0` leaves those rows in the container.

- On this host the R9700 takes 199,851,212 of 320,001,536 rows (62.5%, about 29.8 GiB) and 17.90 GiB go to host
  memory (held but not locked: the account's RLIMIT_MEMLOCK is 11.5 GiB). Both arms run with `--host-pool-mib 24576`
  so the copy fits beside the two ranks' pools.
- Helper selftest: 0 of 960 rows differ across shard boundaries. Standalone the gather is 45 us for one row set and
  191 us at M=2048.
- Served, two XTX, profiled runs (`--profile-ops`, so slower than unprofiled; the arms are comparable with each other):

  | | table on disk (libavx) | R9700 + host memory |
  |---|---|---|
  | decode, 8 short prompts | 23.5 tok/s | 25.3 tok/s |
  | prefill at 2.6K | 203.3 tok/s | 279.0 tok/s |
  | prefill at 7.2K | 639.2 tok/s | 728.3 tok/s |
  | the lookup | 5,642 us a call | 348 us on the device + 24 us from the copy, a gather |

  One start an arm; the disk arm repeated at 23.4 to 23.6 tok/s decode and 638 to 683 tok/s at 7.2K over three runs.
- The load is 91.5 s and is paid inside the first request (the table's address is only known at the first call);
  it is read from the container with sequential reads in 256 MiB pieces - through the cold mapping it stalled for
  over ten minutes.
- Two faults fixed on the way: the row count of a call is the output tensor's, not the `M` parameter's upper bound
  (every decode step had gathered, and written, 32,768 rows); the helper library was refused by the engine as a
  plugin when it sat in `kernels`.
- Unprofiled, one start each (runs `fnp32`, `fnp24`, `fnp20`, `fnp24ng`, `fnp24all`):

  | | decode | prefill 2.6K | prefill 7.2K |
  |---|---|---|---|
  | host pool 32768 MiB a rank, table on disk | 29.5 tok/s | 186.9 | 358.2 |
  | host pool 24576 | 29.8 | 183.5 | 364.3 |
  | host pool 20480 | 30.1 | 207.0 | 401.3 |
  | 24576, `R3_NGRAM_DEV=2` | 33.1 | 393.9 | 597.3 |
  | 24576, `R3_NGRAM_DEV=2 R3_MOE_NATIVE=1 R3_HC_NATIVE=1` | 34.8 | 260.2 | 492.8 |

  The host pool's size changes nothing. The table off the disk is +11% decode and 1.6 to 2.1 times the prefill; the
  two native kernels add another 5% decode (17% over the base). The prefill figures of one configuration differ by
  as much as these arms do from start to start (expert residency at the moment of the request), so only the decode
  column and the disk-against-device prefill step are read as results.
- The profiled disk arms above did 640 to 680 tok/s at 7.2K where the unprofiled ones do 360 to 400: with the
  engine synchronising after every op the prefill is faster, not slower. Not explained; it points at work in flight
  (the mover's transfers, kernels reading experts over the link) contending when the queue runs ahead.

## Flash-Next: radiance's own prefill stager, switched on (2026-10-13, runs `fnsu1`-`fnsu4`)

radiance 1.3.0 has a prefill stager (core/place/stager.h): at a step above the smaller arena levels it copies the next
routed layer's pooled experts onto the card in bulk while the current layer runs, instead of the GEMM reading them
in place over the link. It was idle in every earlier run ("0 staged", no "prefill staging" line): its two buffers are
paid for out of the KV cache, and at one sequence of 32K (271 MiB of KV) the budget cannot fund them. With
`--max-num-seqs 6` (1.6 GiB of KV, about 1.3 GiB of experts off each card) it arms: two 242.88 MiB buffers, 48 routed
layers. Same configuration otherwise (two XTX, wht6, `R3_NGRAM_DEV=2`, host pool 24576 MiB), alternating, unprofiled;
prefill is the server's time to first token (the bench client's long prompts stopped after one token):

| | 2.6K prompt | 7.2K prompt | decode |
|---|---|---|---|
| stager on (`fnsu1`, `fnsu3`) | 4.81 s, 5.52 s (535, 466 tok/s) | 9.93 s, 10.48 s (723, 685 tok/s) | 32.4, 32.5 |
| stager off (`fnsu2`, `fnsu4`) | 9.76 s, 8.61 s (264, 299 tok/s) | 23.24 s, 21.20 s (309, 338 tok/s) | 32.5, 32.5 |

About 1.8 times the prefill at 2.6K and 2.2 times at 7.2K, complete separation, decode unchanged. This is the
mechanism Strata (Niko1221/Strata) uses for long prefill, there with a deeper ring and 8K chunks.

Not yet usable: `fnsu3` hung after its bench on card 0 - a copy kernel (`aql_copy_link`) stopped advancing, the kernel
log shows `sq_intr` errors on 0000:03:00.0, and the server had to be killed. The stager-off runs also logged 5 s
synchronisation waits (three in `fnsu4`) but completed, so the hang is not yet tied to the stager alone.
The hang is a teardown-order bug in the engine, not the stager's copies as such (diagnosis: Ctx::shutdown() drains
the compute streams and frees the activation arena, while the mover's transfer queue is drained only later in
Mover::~Mover; a queued promotion into an arena-lent slot or a stager copy into the arena top then writes freed memory
- sq_intr type 2 is MEMVIOL on gfx11 - and rank 0, the first to shut down, wedges). Without our n-gram helper it
still hung in 2 of 3 runs (`fnh1`-`fnh3`), so the helper is not involved. Engine patch
`engines/radiance/engine-patches/0001-drain-mover-before-arena-free.patch` (built beside the pinned engine by
`tools/lab/radiance/engine-patched-build.sh`) drains each rank's mover before its arena is freed: 3 of 3 runs exit
cleanly (`fnp1`-`fnp3`), no hang. It changes nothing a step computes.
With the patched engine, the stager and `R3_NGRAM_DEV=2` together, 2K against 8K prefill chunks, alternating, all
four runs exiting cleanly (`fnx1`-`fnx4`; 8K is `--max-num-batched-tokens 8192 --checkpoint-interval 8192`):

| chunks | 2.6K prompt | 7.2K prompt | decode |
|---|---|---|---|
| 2048 | 4.36 s, 5.42 s (591, 475 tok/s) | 8.95 s, 10.29 s (802, 697 tok/s) | 32.4 |
| 8192 | 11.83 s, 9.87 s (217, 261 tok/s) | 5.58 s, 5.50 s (1,286, 1,304 tok/s) | 32.1-32.2 |

At 8K chunks a 7.2K prompt is one step that the stager covers, and the cold experts cross the link once for 7.2K
tokens instead of four times: about 1,300 tok/s, against llama.cpp's 1,469 at 6.2K on three cards. A 2.6K prompt at
8K chunks is worse than at 2K: the stager arms only for steps above the next-smaller arena level (4096 tokens here),
so a 2.6K step reads its cold experts in place. The stager's threshold, not the chunk size, is what to fix for short
prompts.
4K chunks fix the short prompt (runs `fny1`-`fny4`, alternating, all clean exits): the stager then arms above 1024
tokens, so a 2.6K step is staged too.

| chunks | 2.6K prompt | 7.2K prompt | decode |
|---|---|---|---|
| 4096 | 2.04 s, 2.09 s (1,262, 1,232 tok/s) | 5.44 s, 5.57 s (1,319, 1,288 tok/s) | 32.3-32.4 |
| 8192, `R3_MOE_NATIVE=1 R3_HC_NATIVE=1` | 9.03 s, 10.65 s (285, 242 tok/s) | 5.18 s, 5.17 s (1,385, 1,388 tok/s) | 34.3 |

So 4096-token chunks (`--max-num-batched-tokens 4096 --checkpoint-interval 4096`) are the setting: about 1,250 to
1,320 tok/s at both lengths, against llama.cpp's 1,012 (3.2K) and 1,469 (6.2K) on three cards. The two native kernels
add 6% decode (32.3 to 34.3); their served quality on Flash-Next is not yet checked (the KL tooling covers the 27B).
Quality on Flash-Next (`tools/lab/radiance/fn-kld.sh`, run `fnk1`; reference: the same two-XTX engine with every
native form off, the table on disk and the exact wire; 42,761 positions):

| arm | KLD mean | top-1 | PPL (ref 10.106) |
|---|---|---|---|
| n-gram rows from the R9700 helper, exact wire | 0.000000 | 100.000% | 10.106 |
| + `--tp-wire wht6` | 0.0746 | 87.18% | 10.436 (+3.3%) |
| + native attention, MoE and HC forms | 0.0751 | 87.52% | 10.451 (+3.4%) |

The n-gram helper is exact and the native forms add nothing measurable, but the lossy wire is not acceptable on
Flash-Next: twenty times its divergence on the 27B (0.0008 over exact there), one top-1 token in eight changed. Every
Flash-Next speed figure above used wht6; the exact wire is to be measured, and is the default for this model.
Without the R9700 (owner: the R9700 is for compute, not storage): the whole n-gram table pinned in host memory
(`--ngram-placement ram`), which fits once the expert pool is smaller - 16384 and 14336 MiB a rank both work with
the stager on. Patched engine, 4K chunks, two XTX only (runs `fnz1`-`fnz5`):

| | wire | 2.6K prompt | 7.2K prompt | decode |
|---|---|---|---|---|
| pool 16384, n-gram in RAM | wht6 | 2.01 s (1,282 tok/s) | 5.34 s (1,343 tok/s) | 32.3 |
| pool 14336 | wht6 | 2.06 s | 5.33 s | 32.3 |
| pool 16384 + built-in MTP draft head (`SPEC=3`) | wht6 | 2.12 s | 5.39 s | 38.2 |
| pool 16384 | exact | 2.15 s (1,198 tok/s) | 5.68 s (1,263 tok/s) | 32.2 |
| pool 16384 + MTP | exact | 2.24 s (1,149 tok/s) | 5.74 s (1,250 tok/s) | 39.3 |

- The n-gram table in RAM is as fast as the R9700 split, so the R9700 is free. Copying the table in takes 200-270 s
  at load.
- The exact wire costs about 5% of prefill on this model, so it is the setting (quality above).
- The container's MTP head lifts decode 22% (32.2 to 39.3). The `fnz3` load logged 21 queue-stall lines while the
  table was being copied, before the server listened; it recovered and served.
- Configuration of record for Flash-Next on two XTX: patched engine, `--tp-wire exact`, `--ngram-placement ram`,
  `--host-pool-mib 16384`, `--max-num-seqs 6`, `--max-num-batched-tokens 4096 --checkpoint-interval 4096`, MTP 3.
MTP against llama.cpp on the same workload (`rad_prompt_bench.py --continue`: the corpus as a raw completion,
greedy, ignore_eos, 256 tokens - as long-ctx-profile.sh timing does; runs `fnm3`, `fnm0`, configuration of record):

| | radiance two XTX | llama.cpp three cards (`lcfn-*`) |
|---|---|---|
| decode at ~3.2K, no draft | 33.0 tok/s | 37.9 |
| decode at ~3.2K, MTP 3 | 53.6 tok/s | 72.2 |
| decode at ~6.5K, no draft | 34.1 tok/s | 36.9 |
| decode at ~6.5K, MTP 3 | 44.8 tok/s | 55.6 |
| draft acceptance | 0.61-0.64 mostly (`--live`) | 0.79 (179/226), 0.56 (160/284) |

Two gaps: the base step is about 10% slower, and acceptance is lower at the shallow depth. Radiance's draft head is
the one in the container; llama.cpp uses a separate Q8_0 MTP GGUF - the head's precision is the first suspect.
Next: a deeper stager and larger prefill steps (`--max-num-batched-tokens 8192`), which raise the
tokens served per byte of cold experts moved.

## Other engines on the RX 7900 XTX (looked up 2026-10-10)

- **llama.cpp, Vulkan backend.** llama.cpp issue #20934: on an RX 7900 XTX, Llama-2 7B Q4_0 decodes at 174.6 tok/s
  through Vulkan against 143.8 through ROCm/HIP, about 20%. The maintainer's explanation: RDNA3 and RDNA4 can run
  packed FP32 math at twice the throughput, the compiler rarely emits it in wave32, and wave64 on RDNA is available
  through Vulkan only, not through HIP; the HIP backend's kernels are the CUDA ones, designed for NVIDIA. An
  experimental `-mwavefrontsize64` HIP build gave +0.2% to +16% on decode and is not supported by AMD. Closed as
  expected behaviour. Bearing on libr3: every kernel here is wave32 and its matrix builtins are the `_w32` forms.
- **MLC-LLM** (Apache TVM Unity, generated kernels, ROCm or Vulkan): its 2023 write-up puts the 7900 XTX at 80% of
  an RTX 4090 on 4-bit Llama-2 single-stream decode. No multi-GPU support.
- **hipfire**: see the section above; single card per model for Flash-Next, mixed architectures opt-in only.
- **vLLM on ROCm** (the owner's earlier gfx1100 deployment, above): AITER/Triton kernels.

None of these is a candidate to replace radiance or llama.cpp here; the two with kernels worth reading for this
card are hipfire and llama.cpp's Vulkan backend.

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
