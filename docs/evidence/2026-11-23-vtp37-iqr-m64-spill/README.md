# VTP37: gfx1100 IQR 4-bit table M64 spill experiments

Result: **not accepted**. The zero-spill requirement at NT2 for both MW4
and MW8 was not met, so the kernel change was not retained.

Only the gfx1100 4-bit table-form IQR narrow kernel with NT >= 2 was altered
in trial build-tree rewrites. The upstream source and gfx1201 provider were
untouched. No unaccepted rewrite is included in this commit.

All four experiments compiled with the pinned HIP toolchain and passed the
component's rewrite-anchor check. The code-object comparison below is from a
single-component, single-unit `r4d_gemm_iqr_p2` build using `R3_NATIVE=ON`
and `R3_ONLY=r4d_gemm_iqr_p2`. The baseline resource record was extracted
from the unchanged component.

- **A** deferred the next weight chunk fetch until after the current WMMA loop.
- **B** also deferred next-chunk activation prefetch.
- **C1** serialized the WMMA instructions per 16-token tile, with the fold left
  after both tiles.
- **C2** ran a complete WMMA-and-fold pass per token tile over the same decoded
  128-K weight block, leaving the source order for other forms unchanged.

Values in the table are `vgpr_count / vgpr_spill_count / private_segment_fixed_size`
(private segment in bytes); U = unchained, C = chained.

| NT | MW | Chain | Stock: VGPR/spill/private | A | B | C1 | C2 |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 4 | U | 240/64/164 | 240/64/164 | 240/64/164 | 240/64/164 | 240/64/164 |
| 1 | 4 | C | 240/62/156 | 240/62/156 | 240/62/156 | 240/62/156 | 240/62/156 |
| 1 | 8 | U | 256/47/96 | 256/47/96 | 256/47/96 | 256/47/96 | 256/47/96 |
| 1 | 8 | C | 256/47/96 | 256/47/96 | 256/47/96 | 256/47/96 | 256/47/96 |
| 2 | 4 | U | 240/357/1052 | 240/381/1092 | 240/417/1156 | 240/412/1140 | 240/401/1168 |
| 2 | 4 | C | 240/359/1060 | 240/383/1100 | 240/419/1164 | 240/414/1148 | 240/403/1176 |
| 2 | 8 | U | 256/298/932 | 256/323/984 | 256/357/1028 | 256/339/1012 | 256/338/1020 |
| 2 | 8 | C | 256/300/940 | 256/325/992 | 256/359/1036 | 256/341/1020 | 256/340/1028 |
| 3 | 4 | U | 240/750/2280 | 240/774/2296 | 240/848/2480 | 240/817/2412 | 240/729/2088 |
| 3 | 4 | C | 240/752/2288 | 240/776/2304 | 240/855/2464 | 240/819/2420 | 240/731/2096 |
| 3 | 8 | U | 256/693/2180 | 256/709/2184 | 256/767/2236 | 256/766/2320 | 256/681/1968 |
| 3 | 8 | C | 256/695/2188 | 256/711/2192 | 256/767/2328 | 256/768/2328 | 256/697/2000 |
| 4 | 4 | U | 256/1020/3132 | 256/1027/3116 | 256/1093/3228 | 256/1077/3064 | 256/1039/2948 |
| 4 | 4 | C | 256/1022/3140 | 256/1029/3124 | 256/1095/3236 | 256/1079/3072 | 256/1034/2916 |
| 4 | 8 | U | 256/970/3140 | 256/978/3232 | 256/1024/3168 | 256/1012/3032 | 256/971/2904 |
| 4 | 8 | C | 256/968/3132 | 256/980/3240 | 256/1026/3176 | 256/1014/3040 | 256/973/2912 |

Of the 144 `iqr_m64_kernel` instantiations in the measured compilation unit,
the other 128 had **identical** resource records in every trial. NT1 also stayed
unchanged. The final C2 whole-unit code-object comparison additionally covered
840 kernel signatures: all 824 non-target resource triples matched stock.

## Gate and test scope

The NT2 zero-spill gate **failed**, including the full C2 fallback. In keeping
with the brief's stop rule, no GPU lease was acquired, no M1-64 bit-for-bit
comparison was run, and no cold timing replay was run. The six drafter shapes
and the ragged N shape were therefore not validated. Full `libr3` plugin build:
**not run**. `r3_selftest` IQR checks: **not run**, so zero FAIL is **not
claimed**. No serving run, ms/step A/B, KLD, or gfx1201 run was performed.

Only unit-targeted compilation and metadata extraction, plus rewrite source
validation, were performed. The separate task report records commands and the
lack of measured speedups. Lead review is required before any further change.
