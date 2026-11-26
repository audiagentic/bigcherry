# IQ6R decode deep-speed investigation

## Stage 1 — kernel map and code-object audit (2026-11-24)

**Status: source dispatch mapped; compiled gfx1100/gfx1201 evidence inspected; live kernel-to-layer correlation and exact head-fixed 2020.8.7 binary match unverified.** A previous profiler census did not capture a successful inference-time symbol trace, so source selection must not be passed off as actual captured dispatch.

### Operation-to-kernel map

Trunk MAD trellis format is **kind 6**, FormT(6,B,0,16,0,3), B=6 or B=7, row-f32 scale; kind 5 is a different stored-table trellis. Both gfx1100 and gfx1201 build corresponding specialisations. Source: the IQR GEMM launcher, the operation registry, and the form-list header.

| Operation family | Operation and launch |
|---|---|
| QKV / SSM input and output projections | gemm_nt_q → gemm_iqr_nt_m64 → r4d_rad_gemm_iqr_m64, ordinary B6/B7 form |
| Attention output and FFN down projections | Same ordinary B6/B7 form; code-object specialization depends on M, N and chain flag, not the operator name |
| FFN gate/up, M≤16 | gemm_nt_q_gated → gemm_iqr_gated → gated IQR GEMV or M16; one launch folds the two gate/up half tiles and fused quantiser |
| FFN gate/up, M=24/32 | The fused row explicitly has M≤16 and cannot serve this case; a separate wide/unfused gate-product path is required, but its exact runtime sequence has not been traced |
| Four-class vocabulary-weighted head | logits_gemm_vocab_iqr → vocab_quant_kernel → vocab_tile_kernel<NT,8,STAGE> → vocab_gather_kernel at M≤32; the four MAD-width classes are descriptors inside one tile kernel, not four separate GEMM launches |
| DFlash2 drafter linears | When using its rotated 4-bit TABLE/BF16 form, FormT(3,4,0,0,0,0), dispatch is IQR GEMV or M16; M17–64 uses separate 16-token slices on the aligned i8a8h route |
| IQ8R control | Aligned i8a8h int8/256-K-group uses gemm_i8_kpar_kernel<4,256,NT,CH>, except when its documented split pin, M/N or alignment guards force a fallback |

For **each** ordinary MAD trellis B6 or B7 form, the source determines the following exact template per M. Here F=FormT(6,B,0,16,0,3), CH is r4dchain or R4dUnchained, KW=4 when N≥4096 and KW=8 otherwise.

| M | Selected device template |
|---:|---|
| 1 | iqr_gemv_kernel<F,1,KW,false,CH> |
| 4 | iqr_m16_kernel<F,KW,false,1,CH> |
| 8 | iqr_m16_kernel<F,KW,false,1,CH> |
| 16 | iqr_m16_kernel<F,KW,false,1,CH> |
| 24 | iqr_m16_kernel<F,KW,false,2,CH> |
| 32 | iqr_m16_kernel<F,KW,false,2,CH> |

M=4/8/16 share an instantiation (actual M is a launch parameter); M=24/32 share the two-token-tile instantiation. The GEMV choice for the B6/B7 row-f32 MAD forms is limited to M=1; their M=2–16 decode route is the WMMA tile. N=13312/6144/5120/6656 use KW4; N=3072/2304/1536 use KW8. The six replay shape pairs listed in the brief are supported. The layer-by-layer link between these shapes and individual dispatch IDs is not verified.

For the **fused gate/up** kernel at M=1, select iqr_gemv_kernel<F,1,KW,true,CH>; at M=4/8/16 select iqr_m16_kernel<F,KW,true,1,CH>, doubling the workgroup wave count over the plain form. For the **head**, M=1/4/8/16 selects vocab_tile_kernel<1,8,true> and M=24/32 selects vocab_tile_kernel<2,8,true>, with preceding quantisation and following gather. For the **4-bit TABLE drafter**, M=1/4/8 selects iqr_gemv_kernel<F,M,KW,false,CH>, M=16 selects iqr_m16_kernel<F,KW,false,1,CH>, and M=24/32 emits two 16-token M16 slices with separate chain packets when the output is aligned. This last claim is about a qualified source route, not proof of which drafter weight form was active in each real call.

### Compiled resources: representative *chained* specialisations

All entries below: VGPR / SGPR / LDS bytes / private bytes; then register-budget wave count floor(1536/VGPR), not active CU waves. B6/B7 are computed MAD trellis, row-f32, KW4 unless stated otherwise.

| Route and M | gfx1100 | gfx1201 |
|---|---|---|
| B6 GEMV M1 | 117 / 31 / 256 / 0; 13 waves | 114 / 28 / 256 / 0; 13 |
| B6 M16 NT1, M4/8/16 | 145 / 70 / 8704 / 0; 10 | 143 / 38 / 8704 / 0; 10 |
| B6 M16 NT2, M24/32 | 176 / 74 / 16896 / 0; 8 | 156 / 48 / 16896 / 0; 9 |
| B7 M16 NT1, M4/8/16 | 147 / 70 / 8704 / 0; 10 | 145 / 38 / 8704 / 0; 10 |
| B7 M16 NT2, M24/32 | 178 / 74 / 16896 / 0; 8 | 158 / 48 / 16896 / 0; 9 |
| B6 fused gate/up M16 NT1 | 144 / 78 / 17412 / 0; 10 | 149 / 46 / 17412 / 0; 10 |
| B6 plain KW8 M16 NT1 | 145 / 70 / 17408 / 0; 10 | compiled, not separately extracted here |
| B6 plain KW8 M16 NT2 | 176 / 74 / 33792 / 0; 8 | compiled, not separately extracted here |
| IQ8R kpar, NT1 M1–16 | 182 / 31 / 4096 / 0; 8 | 147 / 38 / 4096 / 0; 10 |
| IQ8R kpar, NT2 M24/32 | 155 / 31 / 8192 / 0; 9 | 194 / 36 / 8192 / 0; 7 |

The computed trellis M16 NT1/2 samples are **spill-free** on both architectures. The separate gfx1100 B6 iqr_m64_kernel NT2 KW4 chained kernel spills (240 VGPR, 85 SGPR, 17,156 LDS, 1,124 private bytes), but it is not the selected aligned M24/32 computed-trellis path. TABLE B4 drafter KW4 GEMV M1 is 115/37/512/0 on gfx1100, 99/36/512/0 on gfx1201; M16 NT1 is 134/50/8960/0 versus 107/30/8960/0.

**Head version warning:** sampled earlier head code objects show gfx1100 vocab_tile_kernel NT1 240 VGPR / 76 SGPR / 25,856 LDS / 1,280 private bytes and NT2 240/80/34,304/5,576; gfx1201 NT1 is 195/40/25,856/0 and NT2 240/42/34,304/160. These are **pre-head-fix code objects**, not the qualified PR1501 head tile; do not describe those spills as current served head performance.

### Disassembled instruction mix (per *static code body*)

Instruction counts were taken from exact mangled symbols with llvm-objdump, chaining enabled, KW4. Loads include global, scalar and LDS reads. Exchange is **ds_bpermute only**; the total lane-permutation family may be larger. The counts are not hardware-retired instructions.

| Sample | ISA total | Loads | VALU | WMMA | ds_bpermute |
|---|---:|---:|---:|---:|---:|
| gfx1100 B6 GEMV M1 | 799 | 36 | 463 | 0 | 0 |
| gfx1100 B6 M16 NT1 | 1465 | 72 | 958 | 8 | 64 |
| gfx1100 B6 M16 NT2 | 2073 | 97 | 1396 | 16 | 112 |
| gfx1100 B6 gated M16 NT1 | 2416 | 139 | 1445 | 8 | 64 |
| gfx1100 B7 M16 NT1 | 1467 | 74 | 958 | 8 | 64 |
| gfx1100 IQ8R kpar NT1 | 725 | 57 | 308 | 16 | 68 |
| gfx1201 B6 GEMV M1 | 765 | 50 | 452 | 0 | 0 |
| gfx1201 B6 M16 NT1 | 890 | 82 | 510 | 8 | 0 |
| gfx1201 B6 M16 NT2 | 1050 | 113 | 595 | 16 | 0 |
| gfx1201 B6 gated M16 NT1 | 1934 | 170 | 1019 | 8 | 0 |
| gfx1201 IQ8R kpar NT1 | 574 | 86 | 235 | 16 | 0 |

At **nominal** 0.75 payload bytes per B6 weight and 1.0 per IQ8R weight, dividing the static NT1 body by 16×128 logical weights gives gfx1100 1465/(2048×.75)=**0.954 static ISA/payload byte** versus IQ8R 725/2048=**0.354**, 2.69×. The corresponding gfx1201 static quotients are 890/1536=**0.579** versus 574/2048=**0.280**, 2.07×. **This does not answer the requested exact *executed* instructions per weight/byte**: loop execution counts, tile reuse, program prologue, tail paths, scale overhead and effective weights per iteration must be reconstructed and/or profiled. These quotients are explicitly static footprint indicators, not dynamic throughput metrics.

**Finding:** the M4–16 gfx1100 computed-trellis M16 route has much more VALU and permutation/fragment work than its IQ8R comparison, plus additional NT2 state at M24–32. The gfx1201 compiled path removes the ds_bpermute shim and reduces static instruction pressure. B6 GEMV M1 has no WMMA/exchanges on either GPU; its bandwidth deficit must be analysed separately.

### Provenance, blockers and next gate

The IQR and IQ8R dispatch descriptions and head templates above are source-verified. gfx1100 resources/ISA are from a previously extracted **2020.8.5** full-plugin code-object set; gfx1201 resources/ISA are from an independently extracted **2020.8.0** library's native gfx1201 code objects in task scratch. Neither artifact is a proof of byte-for-byte identity with the exact served **2020.8.7 + head-fix** stack. Earlier tracing attempts did not resolve successful live kernel names. No GPU, build, serving, A/B or KLD run was performed in stage 1. Stage 2 must do short leased single-unit cold/prefetch-warm replays on six shapes and M1/4/8/16, both B6/B7 versus IQ8R, calculate an issue-aware cost model and check the version delta. The owner must not interpret this as an exact live kernel trace.


## Stage 2A — cold unit replay and issue/bandwidth cost model (2026-11-24)

**Cold validation complete; prefetch-warm replay and exact current-release binary identity NOT YET qualified.** Six trunk shapes × M=1/4/8/16 × forms B6/B7/IQ8R = **72 successful gfx1100 cold conditions**. Each was the median of 15 launches with an untimed 256-MiB cache flush immediately before each. Runs used short queued GPU leases, one lease per shape, released between. The 13312×5120 shape was repeated in two further independent leased runs: the table reports median-of-three 15-launch medians for that shape; the other five shapes have one 15-launch median each.

The baseline plugin was 2020.8.5 (not proven byte-identical to the 2020.8.7 + head-fix serving release). Reproduction is the selftest --perfiqr N K M 6|7 15 --cold 256 route for the row-f32 computed MAD trellis, and --perfw8 N K M 15 --cold 256 for the IQ8R weight-rotating narrow row. Both use i8a8h rotated activations, with B6/B7 f32 row scales and IQ8R BF16/group256 scales. IQR rotates over more than 256 MiB of code copies; IQ8R rotates over up to 48 int8 copies; both flush 256 MiB. The IQ8R harness also probes unrelated rows before the timed cold row, unlike the IQR-only run, so tiny differences need a strictly matched minimal harness before any change is accepted.

The comparison read target is **768 GB/s**, 80% of the 960-GB/s theoretical plain-read bandwidth. Effective bytes B6 = 0.75 NK + 4N; B7 = 0.875 NK + 4N (codes + one f32 scale per row). IQ8 GB/s divides bare NK by time (omits its scale plane). Percentages below are against that 768-GB/s ceiling, not peak theoretical bandwidth.

| N × K | M | IQ8 µs | B6 µs | B7 µs | B6/IQ8 | B7/IQ8 | effective GB/s IQ8 / B6 / B7 | B6 / 768 ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 13312×5120 | 1 | 88.2 | 80.6 | 86.4 | 0.91× | 0.98× | 773 / 635 / 691 | 83% |
| 13312×5120 | 4 | 89.0 | 162.5 | 154.4 | 1.83× | 1.74× | 766 / 315 / 386 | 41% |
| 13312×5120 | 8 | 90.1 | 145.5 | 163.1 | 1.62× | 1.81× | 757 / 352 / 366 | 46% |
| 13312×5120 | 16 | 91.5 | 143.2 | 161.1 | 1.57× | 1.76× | 745 / 357 / 371 | 47% |
| 5120×6656 | 1 | 47.4 | 48.7 | 51.4 | 1.03× | 1.08× | 719 / 525 / 581 | 68% |
| 5120×6656 | 4 | 47.2 | 97.6 | 97.1 | 2.07× | 2.06× | 722 / 262 / 307 | 34% |
| 5120×6656 | 8 | 47.4 | 97.5 | 102.4 | 2.06× | 2.16× | 719 / 262 / 291 | 34% |
| 5120×6656 | 16 | 47.4 | 97.7 | 65.2 | 2.06× | 1.38× | 719 / 262 / 458 | 34% |
| 6144×5120 | 1 | 43.3 | 42.8 | 46.5 | 0.99× | 1.07× | 726 / 552 / 592 | 72% |
| 6144×5120 | 4 | 43.5 | 56.7 | 57.4 | 1.30× | 1.32× | 723 / 417 / 480 | 54% |
| 6144×5120 | 8 | 43.9 | 55.8 | 86.6 | 1.27× | 1.97× | 717 / 423 / 318 | 55% |
| 6144×5120 | 16 | 44.2 | 86.1 | 92.9 | 1.95× | 2.10× | 712 / 274 / 297 | 36% |
| 3072×5120 | 1 | 25.8 | 28.2 | 29.3 | 1.09× | 1.14× | 610 / 419 / 470 | 55% |
| 3072×5120 | 4 | 26.0 | 52.1 | 48.8 | 2.00× | 1.88× | 605 / 227 / 282 | 30% |
| 3072×5120 | 8 | 26.1 | 51.6 | 48.8 | 1.98× | 1.87× | 603 / 229 / 282 | 30% |
| 3072×5120 | 16 | 26.1 | 51.5 | 48.4 | 1.97× | 1.85× | 603 / 229 / 285 | 30% |
| 5120×2304 | 1 | 22.3 | 24.9 | 27.0 | 1.12× | 1.21× | 529 / 356 / 383 | 46% |
| 5120×2304 | 4 | 22.3 | 44.3 | 48.1 | 1.99× | 2.16× | 529 / 200 / 215 | 26% |
| 5120×2304 | 8 | 22.4 | 48.0 | 47.5 | 2.14× | 2.12× | 527 / 185 / 218 | 24% |
| 5120×2304 | 16 | 22.4 | 44.8 | 46.7 | 2.00× | 2.08× | 527 / 198 / 221 | 26% |
| 5120×1536 | 1 | 17.6 | 22.6 | 22.9 | 1.28× | 1.30× | 447 / 262 / 301 | 34% |
| 5120×1536 | 4 | 17.7 | 34.3 | 34.5 | 1.94× | 1.95× | 444 / 173 / 200 | 22% |
| 5120×1536 | 8 | 17.7 | 34.2 | 26.0 | 1.93× | 1.47× | 444 / 173 / 265 | 23% |
| 5120×1536 | 16 | 17.5 | 34.2 | 35.2 | 1.95× | 2.01× | 449 / 173 / 196 | 23% |

**Measured contrast.** Median B6/IQ8 time ratio across six shapes is **1.06×** at M1 and **1.96× / 1.96× / 1.96×** at M4/8/16. B7 median ratio is **1.11×** at M1 and ~**1.91–1.93×** at M4/8/16. The compression advantage is cancelled at GEMV M1 but reversed strongly in the WMMA tile. Several single-run outliers at M8/16 move significantly when remeasured; do not treat one apparently fast 15-launch median as a durable optimization. An independent test reported in the brief also finds the gfx1201 native B6 faster than IQ8R on the same forms and the gfx1100 M4–16 kernel slower: that is **corroborating supplied data, not another run credited here**.

**Memory floor and issue excess.** For N=13312,K=5120, B6 codes and scales total ~51.17 MB, versus bare IQ8 codes 68.16 MB. B6 M4 took **162.5 µs** or **315 GB/s**, only **41%** of the 80%-read ceiling; ideal transfer at 768 GB/s takes **66.63 µs**, leaving **95.87 µs of excess elapsed time**. IQ8 M4 took **88.95 µs**, or **766 GB/s**, nearly the read ceiling. B6 M1 took **80.6 µs** (635 GB/s); only ~14 µs over its compressed read floor, and still faster than IQ8 M1 at 88.2 µs. The M4 deficit is therefore 2.44× its compressed-byte streaming floor, not an inherent requirement to move more bytes.

**Instruction-level bound.** The MAD trellis has 2^16 states, not merely 64 table entries. Its state advance shifts in the next B-bit code and masks the state; the value is a multiply-add, byte-mask and masked-byte sum, followed by scale folding. GEMV consumes those values with integer dot4 and f32 accumulation. The M16 WMMA route additionally rearranges them into matrix fragments via LDS/transposition, repacks or requantises the activations and executes WMMA through gfx1100 fragment shims. The earlier code-object sample B6 M16 NT1 contains 1465 static ISA instructions (958 VALU, 64 ds_bpermute, 8 WMMA) versus IQ8R kpar 725 (308 VALU, 68 ds_bpermute, 16 WMMA). Each static body includes loops and a prologue; static counts **do not** equal retired operations per weight. The previously supplied source estimate of roughly two basic decode ALU operations per weight is only an arithmetic floor, not an exact compiled instruction count; reformatting, transpose, scale, live-state and issue overhead dominate the observed gap.

Without sampled clocks or instruction counters, exact dynamic cycles/weight by state advance, MAD, scale and WMMA **cannot be stated**. A reproducible elapsed-time CU-cycle-equivalent bound is C_weight = latency_seconds × shader_clock_Hz × CU_count / (N×K). With a merely illustrative 96 CUs at 2.3 GHz, big-shape B6 M4 costs 0.527 aggregate CU-cycles per weight, streaming floor 0.216 and excess 0.311; M1 B6 costs 0.261, excess 0.045. These are CU-normalised wall-clock quantities, **not measured clock, actual SIMD cycles or execution-counter categories**. Low observed throughput and the high code footprint point to **fragment staging/issue/occupancy**, not proven intrinsic state arithmetic throughput saturation. At M1 no WMMA or bpermute occurs and compute-headroom remains.

**Unmet stage-2 gates at this checkpoint:** the specified cold + untimed 12-MiB prefetch-warm replay has not yet run; the post-head-fix serving-release code objects were unavailable; actual issue/stall hardware counters and a live kernel-to-layer trace were not captured. The already supplied gfx1201 B6 93 µs versus IQ8 117 µs (large shape) implies 79% latency and ~87% of its own read ceiling, but is not an independently repeated measurement in this stage.
## Stage 2B — 12-MiB prefetch-warm single-unit replay (2026-11-24)

**Measured 6 of six requested shapes with full B6/B7/IQ8R coverage at M=1/4/8/16: 72 warm conditions passed.**

The scratch-only comparison harness was compiled under nice -n 19 and does not modify the native kernel or installed runtime. For each timed kernel it flushes 256 MiB, then launches the existing registered prefetch operation on the first min(12 MiB, weight-code bytes) of that same rotated weight copy, **without measuring the prefetch**; it measures the subsequent GEMM with device events. Each point is the median of 15 such individual launches; leases were queued, short and released after each shape. The same static 2020.8.5 plugin was used for both cold and warm results, rather than a current-serving-release claim. The IQ8R harness's additional untimed comparison rows, scale layout and copy rotation differ from the IQR harness, so a few-percent advantage is not qualified as a regression decision. No serving run, A/B or KLD was performed.

**Cold vs prefetch-warm timings in microseconds** (B6 and IQ8R plus the B7 warm comparison):

| N × K | M | B6 cold → warm µs | IQ8R cold → warm µs | B7 warm µs | warm B6 / IQ8R |
|---|---:|---:|---:|---:|---:|
| 13312×5120 | 1 | 80.6 → 64.2 | 88.2 → 78.8 | 79.8 | 0.81× |
| 13312×5120 | 4 | 162.5 → 152.5 | 89.0 → 78.5 | 144.2 | 1.94× |
| 13312×5120 | 8 | 145.5 → 147.2 | 90.1 → 78.5 | 154.7 | 1.88× |
| 13312×5120 | 16 | 143.2 → 145.8 | 91.5 → 79.5 | 158.7 | 1.83× |
| 5120×6656 | 1 | 48.7 → 35.4 | 47.4 → 36.3 | 40.6 | 0.98× |
| 5120×6656 | 4 | 97.6 → 92.7 | 47.2 → 36.0 | 97.5 | 2.58× |
| 5120×6656 | 8 | 97.5 → 91.2 | 47.4 → 35.9 | 97.6 | 2.54× |
| 5120×6656 | 16 | 97.7 → 97.5 | 47.4 → 36.0 | 95.0 | 2.71× |
| 6144×5120 | 1 | 42.8 → 36.0 | 43.3 → 32.2 | 37.0 | 1.12× |
| 6144×5120 | 4 | 56.7 → 81.3 | 43.5 → 32.4 | 82.3 | 2.51× |
| 6144×5120 | 8 | 55.8 → 82.0 | 43.9 → 32.7 | 81.9 | 2.51× |
| 6144×5120 | 16 | 86.1 → 80.7 | 44.2 → 33.3 | 82.5 | 2.42× |
| 3072×5120 | 1 | 28.2 → 22.2 | 25.8 → 18.2 | 23.7 | 1.22× |
| 3072×5120 | 4 | 52.1 → 45.0 | 26.0 → 18.0 | 48.6 | 2.50× |
| 3072×5120 | 8 | 51.6 → 44.9 | 26.1 → 17.7 | 44.9 | 2.54× |
| 3072×5120 | 16 | 51.5 → 48.5 | 26.1 → 17.5 | 44.9 | 2.77× |
| 5120×2304 | 1 | 24.9 → 20.6 | 22.3 → 16.8 | 22.0 | 1.23× |
| 5120×2304 | 4 | 44.3 → 44.0 | 22.3 → 17.0 | 43.0 | 2.59× |
| 5120×2304 | 8 | 48.0 → 41.5 | 22.4 → 17.3 | 43.9 | 2.40× |
| 5120×2304 | 16 | 44.8 → 43.5 | 22.4 → 17.0 | 43.7 | 2.56× |
| 5120×1536 | 1 | 22.6 → 18.4 | 17.6 → 15.5 | 18.3 | 1.19× |
| 5120×1536 | 4 | 34.3 → 31.4 | 17.7 → 15.2 | 31.8 | 2.07× |
| 5120×1536 | 8 | 34.2 → 31.5 | 17.7 → 15.3 | 31.8 | 2.06× |
| 5120×1536 | 16 | 34.2 → 30.6 | 17.5 → 15.0 | 29.7 | 2.04× |

**Warm diagnosis.**
- M=1: median across completed shapes of warm B6/IQ8R elapsed-time ratio **1.15×**; observed range 0.81–1.23×.
- M=4: median across completed shapes of warm B6/IQ8R elapsed-time ratio **2.50×**; observed range 1.94–2.59×.
- M=8: median across completed shapes of warm B6/IQ8R elapsed-time ratio **2.45×**; observed range 1.88–2.54×.
- M=16: median across completed shapes of warm B6/IQ8R elapsed-time ratio **2.49×**; observed range 1.83–2.77×.

A cached/prefetched weight-prefix can produce a raw encoded-byte/time quotient above the **768-GB/s cold plain-read target**, because the numerator includes bytes serviced by cache and the untimed prefix was brought into cache deliberately. It is invalid to interpret that quotient as *measured DRAM* GB/s. The unprefetched lower-bound bytes are max(0, B6 total encoded bytes−12 MiB) or corresponding B7/IQ8 code bytes. Kernel startup, active cache eviction, scale-plane traffic and the fraction of prefetched weight still resident are unknown; their contribution cannot be separately inferred from the elapsed time. The warm result tests **whether a cache head start removes the gap**, not hardware read throughput after the prefetch.

A prefetch does **not guarantee shorter B6 latency**: for 6144×5120 M4 the cold single median is 56.7 µs but the warm median is 81.3 µs; M8 behaves similarly. The baseline cold repetitions at the large shape also varied, so these disparities are flagged as run-to-run/timing-method instability rather than alleged negative cache bandwidth or a code regression. The issue/fragment result is robust because the warm B6 time still loses to the simultaneously measured warm IQ8R on all M≥4 shapes.

The observed gap at M≥4 persists despite prefetch in every completed shape. Therefore the tested gfx1100 M16 path cannot be explained purely by weight-memory latency; improving issue/dataflow/fragment staging and lowering live register pressure remains the first-ranked route. M=1 has a qualitatively different no-WMMA issue and frequently approaches or beats IQ8R on large shapes. The result does **not** prove that static lane exchanges alone are responsible, nor does it produce instruction-retirement or clock counters. The exact served 2020.8.7+head-fix binary remains unverified.

**All six requested warm shape matrices completed with exit code 0.** The stage-2 timing design gate is complete for this scratch plugin; serving-version identity and hardware issue counters remain outstanding.


## Stage 3 — ranked design and conditional projections (2026-11-24)

**Disposition: design only, no implementation.** This stage uses the independent six-shape cold replay, the compiled disassembly/resource records, the supplied served trace, and the separately supplied native gfx1201 comparison. The complete six-shape cold and 12-MiB-prefetch-warm replay matrices are now measured on gfx1100 (144 timing conditions), but the exact served 2020.8.7 code-object/dispatch identity and retired-issue/clock counters remain unverified.

### Reconciliation before choosing a code path

The two gfx1100 failure mechanisms differ. GEMV M=1 is a no-WMMA dot4 kernel: at the largest shape its 635-GB/s encoded-weight rate is 83% of the 768-GB/s read ceiling and the compressed bytes already give a 0.91× IQ8R elapsed-time ratio. In contrast M=4–16 enters the M16 WMMA tile: its ~315-GB/s B6 read-equivalent rate at the largest shape M4 is just 41% of the ceiling. The static code body is ~2× IQ8R kpar instruction count and has a nontrivial fragment/transpose path. The prior experiment simplifying a lane-exchange instruction found no timing change, so **counting lane exchanges alone is not an adequate performance model**: issue slots, register live ranges, dataflow, scalar synchronisation and active wave occupancy must be checked together.

On gfx1201, native fragments have no corresponding ds_bpermute shim and the supplied B6 M≤8 timing of 93 against 117 µs IQ8R is already within ordinary read-bandwidth expectations; do not impose a gfx1100-specific rewrite on gfx1201. Serving at M24 has an additional **version/dispatch ambiguity**: the source analysed in stage 1 selects spill-free M16 NT2 for an aligned computed trellis, while the served step trace reported a spilling wide-tile route. Inspect the exact deployed 2020.8.7 branch, real output alignment and the live emitted code-object ID before assuming either path is active. A two-slice fallback is useful only when this version check confirms the spilling path.

For computed MAD B6/B7 the 16-bit trellis state is the transition state; a compact 64-value-per-state shortcut is incorrect. There are 65,536 possible states. A direct 8-bit state-to-value table is at least 64 KiB per form even before LDS staging/alignment, occupying an entire nominal 64-KiB CU LDS store if copied per workgroup; a state×incoming-code lookup grows to 4–8 MiB for B6/B7. It would introduce scattered loads and damage occupancy. Do not pursue a blanket LDS LUT without evidence of locality or a smaller algebraic decomposition.

### Derivation of a multi-row dot4 crossover (a candidate to *measure*, not a constant)

The gfx1100 B6 GEMV M1 code body has 65 static dot4 instructions among 463 static VALU instructions. An optimistic weight-decode-sharing model, **assuming that only the dot4 contribution scales with token rows and all other issued work stays fixed**, has relative incremental cost r = 65 / 463 = 0.1404 per extra token. Let T1 be the measured B6 GEMV M1 time and Ttile4 the measured B6 M16-tile M4 time on the same (N,K). Then T_shared(M) = T1 × [1 + r×(M−1)], with the rough breakeven M* = 1 + (Ttile4/T1−1)/r. Using the six cold shapes gives M* approximately **8.2, 8.1, 3.3, 7.0, 6.6, 4.7** respectively. This is a **3–8-row *experimental band***, not a validated dispatch point: static dot4 fraction is not retired instruction cost, added accumulator registers may reduce occupancy, and the tile time changes with M. The release rule must be generated from independently measured per-shape curves and a derived wave/register guard, never a hand-picked M cutoff. Benchmark integer M=2..16 on every shape, comparing both routes' cold and warm timing with identical inputs. Guard by compile-time architecture, form and register budget and select from a derived measured model only if bit identity or the approved FP64 numeric gate passes.

### Ranked candidate changes

**1. gfx1100 M16 fragment pipeline + native accumulation, for MAD trellis B6/B7 (highest likely gain at M4–16; substantial at M24 if M16 NT2 executes).** Take the already separately developed native accumulator path as the starting point, after its review: keep int32 WMMA sums in native accumulator layout across the K scale group and make one f32/bf16 conversion per group. Then change the activations/decoded weights' fragment staging to avoid a register-permuted legacy shim: stage compact sub-tiles through LDS, let KW waves each own a bounded K slab, and use native fragment mapping so no wave accumulates the whole transpose in registers. The 6/7-bit state hashing should execute once per weight tile, not once per token. Derive the guard from available per-SIMD VGPR file, registers per wave, LDS and tile geometry. It must preserve or improve every code object's VGPR, spill bytes and wave budget; avoid changing kinds 5 or TABLE forms. Static upper opportunity is the gfx1100 M16 B6 body gap of 740 ISA instructions against IQ8R (1465 vs 725), not a claim that all 740 are removable or dynamically amortised. The per-large-kernel excess above compressed-bandwidth floor is ~96 µs at M4; *all* of it is an optimistic upper bound, not an achievable gain. Gate: 404-M16-specialisation full resource table, every M1..64 including N=5376 and chained/unchained, BF16 byte identity or explicit FP64 numeric gate, actual iqr selftests and cold+warm no regression. Coordinate with the concurrent native-accumulator work rather than overwriting its rewrite.

**2. Reuse decoded states in a multi-row non-WMMA dot4 GEMV-style tile at M2–8 (medium-to-high payoff, less certainty).** Vectorise state-to-value MAD once per 16×128 code unit and keep decoded int8 words shared over a small register-resident set of token rows; dot4 each row's activations against the shared values, fold the f32 row scale in the established order, and reduce/emit once. Use a per-form constexpr occupancy limit to keep M×accumulator state from crowding out other waves. First compare at M2,3,4,6,8,12,16 against the existing M16 tile. The optimistic crossover span derived above is 3–8, to be *measured* rather than coded as a literal. An optimistic M4 big-shape estimate from T1=80.6 µs and r is 115 µs, versus the measured tile 162.5 µs: potential ~47 µs per call but **still slower** than IQ8R 89 µs; reaching the 66.6-µs encoded-byte roof would also require memory/issue improvements. Preserve exact state advance and fma/bias rounding. Gate: identical B6/B7 decoded values and outputs for all M1..64, chained/unchained, ragged N and random/edge code streams; full resource/measurement gates.

**3. Occupancy / load concurrency tuning for GEMV M1 (moderate confidence, smaller absolute saving).** Keep the existing dot4 and B6/B7 state recurrence; reduce live state or stage two independent K groups to overlap global loads with scalar/value arithmetic without adding a per-row weight reload. At big-shape M1 the 80.6-µs observed latency has ~14 µs headroom over the 66.6-µs compressed read roof. This can at most trim about 17% of that single call; across the ~7-GB compressed weights/rank, 635→768 GB/s implies at most roughly **1.9 ms** of rank-critical-path headroom if all the long linears are improved and the rank remains the bottleneck. This is a **best-case bound**. Use constexpr register/occupancy derivative, not a hand-tuned prefetch distance, and cold+warm replays. Require bit identity.

**4. 17–64 token spill-path removal / two-16-token slicing (large potential at M24, but blocked on dispatch proof).** If the exact served 2020.8.7 code at M17–64 really dispatches the spilling iqr_m64_kernel, route computed-trellis row-f32 B6/B7 to the independently chained 16-token slices already used by the TABLE B4 variant, or extend the existing M16 NT2/3/4 form where its per-specialisation register guard permits. Before changing code, compare its present emitted function ID with the source-claimed M16 NT2 path; if that path already executes, this lever has zero benefit and the old wide-spill trace needs explanation. Preserve one chain packet per launch, K fold order and output alignment; an M17..64 slice may not be BF16-bit-identical with M≤16. Gate: an explicit FP64/reference numerical qualification on all M1..64, both chain choices and N=5376, with max relative/absolute error and ULP distribution; resource, selftest and cold+warm no-regression, followed by serving review elsewhere. The 24-row trunk gap is 23.2−14.1=9.1 ms, an *upper bound* for all fixes combined, not solely the spill penalty.

**5. Separate LDS decoded-value staging / packed math algebra (exploratory).** If native fragment staging alone is still issue-limited, pack state-window MAD computation across several values using vector bit masks and byte dot instructions, and stage only a narrow 16×K slab with multiple waves sharing it. The target is to reduce repeated state extraction and out-of-order i8 requantisation without increasing LDS above the resource-derived occupancy threshold. This is a kernel dataflow redesign, not a contained one-line patch. Compare static and *dynamic* issue counts and register occupancy. Correctness must cover tail-biting boundaries every 256 rows, bias seed path, and B6/B7 separately.

**6. Gated FFN fusion, head and table lookup (lower priority / not first implementation).** The existing gated GEMV/M16 row already combines gate and up and requantises once at M≤16; treating this as two independent GEMM launches and proposing to fuse them again is a false opportunity. The head was separately fixed and must use the current head-fix release for further profiling. The 64-KiB-or-larger naive state LUT has poor LDS economics; do not build it absent locality data. Preserve the already-fast gfx1201 native path by a gfx1100-only guard.

### Static instruction-footprint budgets per candidate (NOT retired counters)

Each value in this table divides **one compiled code body's static instruction count** by the nominal 16×128 B6 code payload of **1536 bytes** (twice that for an NT2 body). It is an *ISA-footprint density*, not instructions executed per byte or a cycles/weight estimate. The B6 plain M16 NT1 baseline is 1465/1536 = **0.954 static ISA/B** and the same-length IQ8R kpar body is 725/2048 = **0.354 static ISA/B** of its *own* payload (the two compilers/layouts are not equal-work instruction sequences). The comparable gfx1201 B6 M16 quotient is 890/1536 = 0.579. A code-body delta is only a candidate for an issue-time win when the generated loop and occupancy records are measured.

| Lever | Static footprint observation or bound | Dynamic uncertainty and acceptance gate |
|---|---|---|
| Native fragments/accumulation (1) | The gfx1100 B6 M16 body is 0.954 ISA/B; a same-work IQ8R body contains 740 fewer static instructions (740/1536 = 0.482 static ISA per B6-code byte). This is **an observed gap, not a guaranteed removable instruction count**. | Count compiled body and loop-aware issue; zero new register spills/wave loss across all objects; byte identity at every required M |
| Shared dot4 M2–8 (2) | B6 GEMV M1 body is 799/1536 = **0.520 ISA/B**, versus B6 M16's 0.954. The GEMV body handles a different number of output tokens and workgroups; the reduction cannot be asserted as actual throughput savings. | Measure actual M2..16 crossover; prove decoded recurrence and BF16 result identity (else explicit FP64 numeric gate) |
| GEMV occupancy (3) | Starting footprint ~0.520 static ISA/B, with no proven per-byte instruction saving; speed upper bound at the largest shape M1 is the ~14-µs gap to encoded-weight read floor. | Resource-derived concurrency changes only; bit identity plus cold/warm no regression |
| M24–64 wide spill removal (4) | **Only if the deployed wide route is real**: compiled gfx1100 B6 iqr_m64 NT2 body is 4743 static ISA / 3072 nominal B6 bytes = **1.544 ISA/B**, 224 bpermutes and 1124 private bytes. The source-selected iqr_m16 NT2 object is 2073/3072 = **0.675 ISA/B**, 112 bpermutes and zero private bytes. Static body difference **2670 instructions**; source/route mismatch makes it invalid to project speed from this alone. | Confirm live M24 code-object ID; then FP64-qualified M1..64, alignment, chain and selftests; if already M16, this lever has zero opportunity |
| Packed MAD/LDS staging (5) | 16-bit state recurrence and at least multiply-add/mask/byte-sum per value remain; static saving is **unmeasured** pending a prototype, not zero or a made-up point estimate. | Require edge-state/tail-biting bit identity and full compiled resource sweep |
| Gated fusion / lookup (6) | Gate/up already shares one fused kernel below M17: **zero second-launch saving**. A full direct LUT would require 64 KiB per MAD form (or 4–8 MiB for state×code transitions), not less ALU for free. | Do not implement absent a measured reuse/occupancy justification and strict numerical gate |

### Conditional predicted decoder times (NOT new serving A/B)

A step trace supplied with the brief gives gfx1100 trunk **M4 = 15.6 ms IQ6R vs 11.3 ms IQ8R**, **M16 = 16.8 vs 12.0**, and **M24 = 23.2 vs 14.1**. Linear interpolation in M from 4 to 16 gives an **illustrative** M8 trunk 16.0 vs 11.5 ms. Separately supplied full decoder-step IQ6R baselines are **25.8 ms at one stream** and **39.1 ms at eight streams**; these are not the same measurement as the isolated trunk trace, and no M24 full-step timing was supplied. The table therefore labels 1/8 as *full-step* conditional projections and 24 as *trunk-only*; values in different columns must not be added or compared directly.

| Change if it alone passes all gates | M1 full step, ms (base 25.8) | M8 full step, ms (base 39.1) | M24 trunk-only, ms (base 23.2) | How bound derived |
|---|---:|---:|---:|---|
| 1. Native fragment and accumulator | 25.8 | **34.6–37.0** | **18.7–23.2** | Assume half to all of the estimated 4.5-ms 8-row trunk gap, and 0–half the 9.1-ms 24-row trunk gap; not a claim of measured win |
| 2. Shared multi-row dot4 route | 25.8 | **37.0–39.1** | 23.2 | Cap potential at ~half the M8 gap; only serves small row counts |
| 3. GEMV M1 occupancy | **23.9–25.8** | 39.1 | 23.2 | Bound by 635→768-GB/s improvement over compressed rank weights; whether XTX remains critical is unknown |
| 4. Confirmed wide-spill removal | 25.8 | 39.1 | **14.1–23.2** | Zero if source already dispatches M16; at most the full *observed* M24 trunk gap if current path spills |
| 5–6. LDS algebra and gated epilogue | not estimated | not estimated | not estimated | No validated cost separation yet; giving a point estimate would be invented |

These are **scenario bounds for prioritising experiments**, not forecast regressions/wins. In particular, the lower bound of 34.6 ms at M8 still exceeds the 31.7-ms IQ8R full-step baseline: kernel fixes alone may leave head, nontrunk and inter-rank effects. Do not claim a 22% whole-serving reduction merely because compressed weight bytes are 22% smaller.

### Implementation decision and required concrete patch sequence

**Do not implement stage 4 in this worktree now.** The highest-ranked fix is not a contained safe rewrite: it overlaps the concurrent native-accumulation work, needs a coupled fragment/LDS layout change, and cannot satisfy the mandatory *all-kernel* resource table plus exact M1–64 / N5376 / chained and unchained byte or numeric gates from the available baseline binary. The serving-release M24 dispatch discrepancy and the fact that a completed warm12 replay still shows large M4–16 deficits are independent blockers to a trivial change; the missing version-matched tests apply to any new implementation.

A follow-up implementation proposal, **after incorporating the reviewed native-accumulator patch**, is:
1. Record the exact deployed code-object IDs for M=1/4/8/16/24/32 B6 and B7, fused gate/up, head and IQ8R, including output pointer alignment and chain flag. Extract the matching full resource baseline and full code-object set from that release.
2. Add one managed native rewrite in the IQR GEMM source (one patch ID, tests, no vendor edits outside patch) that introduces a gfx1100 B6/B7 native-fragment/LDS-staging helper and per-instantiation constexpr guard derived from the 1536-VGPR wavefile and LDS occupancy; leave gfx1201, stored-trellis kind5 and TABLE forms as shipped. Prove resource non-regression for every compiled kernel, not only the happy path.
3. In a separate reviewable experiment, add the shared-decoded-weight M-row dot4 route and an empirical cross-over selector derived from measured issue and resource geometry; never encode 4 or 8 as an unexplained switch constant.
4. If and only if the 2020.8.7 M24 trace confirms a spilling wide-tile dispatch, extend the qualified chain-safe M16 slicing to computed MAD B6/B7, with a documented FP64 error gate.
5. For each change, first test every M=1..64 and both chain settings on all six real shapes and N=5376, tail-biting boundaries and gate bias seed; reject byte divergence unless an approved FP64 numeric qualification applies. Run r3_selftest IQR suite and cold+12MiB-prefetch-warm replay, check zero new spill/wave loss, and provide a separate no-regression serving gate for the lead to run. Commit locally, do not push or ship.

**Review caveat:** an independent delegated review was unavailable under the tool policy, so no external critique is claimed. Source and compiled code-object disassembly, arithmetic checks and the patch diff were reviewed read-only. This is a technical self-review, not an external sign-off.

## Continuation — 6-bit path optimization ledger (2026-11-25)

The current comparison baseline is the merged accumulate-once/native-fragment M16 tile, not the older 2020.8.5 replay binary. The statuses below distinguish a derivation from tested code.

| Item | Scope | Status / remaining gate |
|---:|---|---|
| 1 | 16-row transpose-free/transposed-product tile | Mapping verified by asymmetric CPU test; no viable resource-neutral candidate; GPU gates and current-native replay remain open |
| 2 | M1 computed-trellis GEMV bandwidth, concurrency and two-K-group overlap | Source/resource audit: two-ahead register risk; no GPU-qualified candidate |
| 3 | NT3/NT4 and 100 previously excluded native-accumulate symbols | All 100 identified; NT3/4 prototype built, but 34 variants lost a wave — rejected |
| 4 | Vocabulary head NT3/NT4 spilled-register residue | Short-lookahead compiled: NT3 35→9 and NT4 797→663 spills; resource-qualified only, GPU gates open |
| 5 | Trellis token-embedding gather per step | All 17 kernels compiled; ten rotated forms save ten barriers and 25–32 instructions; GPU gates open |
| 6 | Shared activation requantization per block | Already once per token/K128 block on traced trunk/head/gated paths; no code change |
| Report only | Trellis MoE kernels | Separate dot-run and LDS/WMMA mapped-run paths documented; unchanged |

### Item 1A — operand-layout derivation, before prototype or GPU run

**Matrix identity and existing orientation.** The GEMM is `Y[m,n] = sum_k X[m,k] * W[n,k]`. The shipped narrow native tile *already* forms the transposed product `D[n,m] = W[n,k] * X^T[k,m]`: native WMMA A=weight, B=activation, and the output epilogue stores `D[n,m]` at `Y[m,n]`. Merely swapping the names of A and B or calling the result transposed cannot remove a transpose.

**Native i8 layout.** For wave lane `l=16h+c`, `h=0..1`, `c=0..15`, a 16x16x16 gfx11 WMMA A fragment expects the 16 consecutive int8s `A[c,16s+0..15]`; B expects `B[16s+0..15,c]`. The native C accumulator slots `t=0..7` map to `D[8h+t,c]`. Both lane halves read the *same* 16 weight/activation K bytes in the current native operand path. Accordingly `A[c,k]=W[n0+c,k]`, `B[k,c]=X[m0+c,k]`, and an accumulator slot is one weight row `n0+8h+t` by token `m0+c`. Per-K128 integer sums, the row-f32 bias-seeded fold, per-wave reduction and output rounding must retain the old order to qualify byte identity.

**Compressed decoder layout and the exact crossing.** Within one 16-output-row x 128-K unit, decoder lane `l` holds the four adjacent K bytes for *every* row: `w_l[r] = pack4(W[n0+r,128kb+4l+0..3])`, `r=0..15`. The trellis decoder reverses internal row `q` to `r=15-q` before writing `w[r]`. Thus 32 lanes x 16 dwords/lane x 4 bytes/dword = **2048 decoded bytes/unit**. A native weight A fragment at K16 step `s=0..7` for row `c` needs exactly `w_(4s)[c]`, `w_(4s+1)[c]`, `w_(4s+2)[c]`, `w_(4s+3)[c]` in that order. Four *different decoder lanes* own the dwords, so an across-lane four-to-one regrouping is unavoidable with the existing packed code layout and recurrence.

The current transpose uses one wave-private LDS tile of `16 * (128+8) = 2176` bytes (8-byte bank skew per row), writes the 2048 decoder output bytes in K-striped lane order, and reads native row-contiguous K16 fragments. Its LDS cost is `2176 * NW` bytes of allocation per block, plus independently allocated reduction scratch. The new native-fragment version removes gfx11 operand-shim exchanges and keeps native int32 sums, **but not this decoder-to-WMMA LDS regrouping**.

**Reversing the product is not transpose-free.** Swapping to A=activation (`X`) and B=weight transposed (`W^T`) produces `Y` rather than `D`, but its B lane still requests sixteen consecutive K bytes for *one* weight channel. Those bytes belong to the same four decoder lanes above. It therefore still needs LDS, a register-wave gather, or different code packing/decoder ownership; its accumulator-to-output map changes as well. A pure register gather trades 16x128 LDS writes/reads for source-lane exchanges, larger live decoded rows and cross-lane scheduling. A decoder reassigned one weight row per lane would need four code-column streams per row and lose the current amortisation of trellis state extraction over all sixteen rows; it cannot be assumed faster without a source/layout experiment.

**Item 1 experiment specification.** Compare (A) the current native LDS transpose, (B) a wave-register K16 gather preserving the exact four-dword ordering, and (C) an A/B-swapped transposed-output arrangement with the same K16 decoded bytes and correct output scatter. Prototype on computed MAD kind6 B6 and B7, row-f32, NT1/NT2, gfx11 only. No speculative hard-coded cutoff: any release guard must follow resource records per instantiation. Hard stops: any new spill or lost wave among changed symbols; any integer-sum or BF16 byte divergence (including bias seed, tail-biting boundaries, M1..64, N=5376, chained and unchained). Run full IQR selftests; only after byte identity, compare six real shapes at M=4/8/16/24/32 with cold 256-MiB flush and untimed 12-MiB prefetch-warm replay versus the **current** native tile and int8. **No candidate, compiled resource table, identity result or timing is claimed at this checkpoint.**

### Item 1B — operand-map verification and disposition

A scratch-only asymmetric integer test reconstructed the 16x128 decode unit from 32 lanes x 16 four-byte packs, gathered each native 16-byte K fragment from four source lanes, and verified both the shipped `W X^T` accumulator addressing and the operand-swapped `X W^T` addressing against 256 exact int32 reference outputs. **Result: PASS, 256/256.** This is a pure mapping test, not a hardware WMMA bit-identity test, not a compiled candidate, and not a latency benchmark.

The tested mapping proves that swapping the product does **not** remove the four-source regrouping for the existing code-stream format. The only alternatives identified are: (1) wave-register cross-lane transpose, with at least the four source dwords for each target fragment and a potentially large shuffle/reg cost; or (2) a new native-row-oriented code layout with different trellis state walks, requiring a conversion-format and decoder change. Neither is a contained, demonstrated resource-neutral rewrite. **Disposition: no release candidate for item 1; do not replace the shipped native LDS tile without an actual resource/identity/performance winner.** In particular, no current-native-vs-int8 cold/warm replay was run in this checkpoint, so its timing requirement remains open rather than fabricated. The old replay values in stage 2 are *not* current-native control timings.

### Item 2 — M1 GEMV occupancy, bandwidth and two-group lookahead audit

The M1 computed MAD GEMV **already pipelines a full 128-K group**: `cur=D::fetch(wave)`, and at the top of each K loop it obtains `nxt=D::fetch(kb+KW)` before `D::hashes(cur)` and the dot4/f32 folds. The launch selects KW=4 at N>=4096 and KW=8 for narrower matrices. For the 13312-output-row case this is 832 workgroups and 3328 launched waves at KW4. Raising KW changes the floating reduction grouping, so cannot be accepted by byte-identity assumption.

With 6-bit MAD L16, each lane's code record is ND=64*6/32=12 dwords of the current unit, plus four continuation dwords from the preceding trellis row: **16 dwords** per extra in-flight record. With 7-bit, ND=14 and the record is **18 dwords**. An actual two-*ahead* load at the top of the loop needs `cur`, `nxt`, and `ahead` simultaneously while the current 64 MAD hashes and subsequent dot instructions consume `cur`; it is not free latency hiding. The compiled previous-release, source-equivalent M1 resource records for plain kind6 row-f32 B6/B7 are below (KW4 and KW8 share these registers); values are *not* a newly built candidate measurement:

| Form / chain | Existing VGPR | SGPR | Spills / private B | 1536/VGPR wave floor | Estimated additional full Raw | Wave floor if added live registers survive |
|---|---:|---:|---|---:|---:|---:|
| MAD B6 chained | 117 | 31 | 0 / 0 | 13 | 16 | 11 |
| MAD B6 unchained | 112 | 28 | 0 / 0 | 13 | 16 | 12 |
| MAD B7 chained | 119 | 31 | 0 / 0 | 12 | 18 | 11 |
| MAD B7 unchained | 114 | 28 | 0 / 0 | 13 | 18 | 11 |

The last column is **an upper-live-state projection, not compiler allocated resource**. A viable two-group rewrite must reclaim 15/10/9/14 live VGPRs respectively to keep the same floor; the guard must be derived per specialization from compiled registers and the wavefile, not a model- or M-specific numerical cutoff. The prior six-shape 12-MiB-prefix prefetch replay measured the big B6 M1 cold 80.6 us and warm 64.2 us, against a 51.17-MB compressed representation and an illustrative cold read floor of 66.6 us. Thus a cold-cache improvement ceiling is only about 14 us at that shape under the assumed bandwidth, and the warm result cannot be treated as DRAM bandwidth. Neither number is a measurement of the current native build; its control replay remains to be run.

**Disposition:** no register-neutral two-group-overlap code is qualified or enabled in this item. Blindly adding `ahead` is predicted to fail the zero-lost-waves resource gate; changing KW is a bit-identity risk. Current-native full-resource rebuild and GPU lease are outstanding, so this is a source/resource feasibility result rather than a successful prototype or completed cold/warm performance gate. Preserve the shipped M1 path until a compiled variant both restores the resident waves and passes the M1 byte gate plus the six-shape cold/warm replay.

### Item 3 — long native M16 token tiles and 100 excluded specializations

The merged native guard enables 304 of 404 M16 instantiations. Kind6 computed MAD accounts for the 100 exclusions: **88 NT3/NT4** (44 of each), **4** kind6 B3 row-f32 NT1 chained, and **8** kind6 B6 NT1 gated+unchained. The B6/B7 ungated NT1/NT2 decode configurations that the target uses are already native. The exclusions are not all proven regressions: the earlier **unrestricted** accumulation build found 46 regressed symbols, whereas the guard conservatively kept 100 stock. To enable the remaining symbols requires an actual current-native per-symbol resource comparison, not a blanket guard removal.

For an NT>=3 native M16 unit, the temporary activation ring is two sets of 8x16-byte fragments (`bq[2][8]`: 64 int32 lane registers), and decoded native A is eight 16-byte fragments (`a11[8]`: 32 registers), on top of 8-float accumulator rows per token, folded correction and trellis `Raw`. A contained **experimental** register-lifetime rewrite can use just one 8-fragment B buffer, loading the current token at use, and stream each A fragment from the existing wave-private transpose LDS plane at its K16 WMMA step rather than retaining eight A fragments simultaneously. It exchanges at most 64 potentially live int32 registers for more local reads and the loss of token lookahead; this is a resource candidate, *not* evidence of a speed improvement. The LDS plane cannot be overwritten before all its streaming consumers finish, so a wave-local barrier/fence is mandatory at the K128 boundary.

The historical unrestricted-guard table illustrates the need for a gate: B5 NT3 chained was 194 -> 221 VGPR (7 -> 6 floor waves); B5 NT4 chained 209 -> 256 (7 -> 6); B7 NT3 chained 216 -> 238 (7 -> 6). Any new native variant must compare **every** one of the 404 M16 records against the currently compiled native/fragments baseline, retain zero new spills, no reduction of floor(1536/VGPR), and no worse LDS, including both chain modes and ragged N=5376. The 12 NT1 exceptions need separate bias-seed and gated-epilogue register records, not a widened NT-only condition. A new constexpr guard must be derived from those records and physical register/wave geometry, with no copied shape cutoffs.

**Disposition:** the source-level stream/reuse candidate and comparison harness have been prepared in scratch only; there is **no compiled candidate resource result or GPU identity** and no shipping rewrite. Current GPU access is reserved by another session; none of these excluded specializations has passed the mandatory byte identity, IQR selftests or cold/warm timing. Leave the existing 304/100 split unchanged, with the source and resource follow-up explicit for the next qualification session.

### Item 4 — vocabulary head NT3/NT4 spill residue

The computed-trellis head is a **separate** kernel family from the trunk M16 tile. Its already-merged native change accumulates integer sums through both int8 activation planes (hi scaled by 128, then lo) and converts the native accumulator once per K128 unit; it does not remove the B6/B7 MAD decoder or the two activation staging planes. The prior gfx1100 **native head** resource scan shows four `vocab_tile_kernel<NT,8,STAGE>` entries (these are recorded scan results, not fresh rebuilt results):

| NT | STAGE | VGPR | VGPR spills | Private B | LDS B | floor(1536/VGPR) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | staged | 167 | 0 | 0 | 25856 | 9 |
| 2 | staged | 205 | 0 | 0 | 34304 | 7 |
| 3 | direct | 240 | **35** | **52** | 42752 | 6 |
| 4 | direct | 256 | **797** | **204** | 51200 | 6 |

Per wave lane, the activation staging ring consists of hi/lo `Stage8 ph[kBP],pl[kBP]`, eight 32-bit values each: with FK256, MW8, kRPP32, `kBP=ceil(16*NT/32)`, so it grows from 16 registers at NT1/2 to 32 at NT3/4. The two decoded-unit prefetch arrays `pf[2],pfn[2]` carry four live records of ND+4 dwords: 64 for B6, **72 for B7**. Native `raw11[NT]` and float `acc[NT]` add 16*NT accumulator values (48 at NT3, 64 at NT4), before the wave decoder, row factors, per-token scale, pointers and branch state. These overlapping live arrays explain why a pure WMMA shim removal did not cure the large-tile spill.

**Candidate 4A:** for only resource-qualified long token tiles, fetch the current pair of K128 weights at the start of each 256-K iteration and eliminate `pfn[2]`; this shortens the per-lane lifetime by up to 32/36 dwords at B6/B7, without changing either integer accumulation order or the f32 fold. It also removes global-memory lookahead across the 256-K boundary: faster code is **not established** even if the compiler removes spills. **Candidate 4B:** split NT4 into two NT2 passes, which re-decodes every code and increases global reads; this is a fallback only if a measured cold/warm replay shows the spill elimination outweighs the duplicated decoder. The serving head uses NT3 at M33–48 and NT4 above M48, so neither candidate may slow those rows. The resource gate must cover all head kernels and preserve NT1/NT2 unchanged.

**Disposition:** quantified the NT3/NT4 spill and live ranges; no edit to the head kernel, no fresh candidate compile, no head byte-identity test or performance run. Do not enable a shorter prefetch or a slice solely because it reduces private bytes: the weight read/issue regression gate remains open while GPU access is reserved.

### Item 5 — trellis-coded token-embedding gather per step

**Scope clarification from the dispatch and built forms.** `embed_lookup_iqr` is a one-CTA-per-(token,128-element) gather over the **row-major compressed embedding table**, not the trunk's computed MAD M16 GEMM. The embedding form list contains rotated `kTrellis` (lookup-table kind 5, B2–B7, L12, E4M3 tensor scale), not `kTrellisMad` kind 6, and also table and zero forms. A run is 256 rows with tail-biting state; for L12 B6/B7 each output thread reconstructs two code-row contributions (`ceil(L/B)=2`) to make its 12-bit state, loads one int8 table entry, computes the scaled value, and participates in a 128-wide inverse Hadamard. One token of width 5120 launches 40 independent CTAs, rather than a 40-fold sequential kernel loop. **Actual step share is unmeasured**; absent a kernel trace do not claim this gather is the serving critical path or that 40 CTA costs can be added serially.

**Exchange cost.** The shipped rotated path puts 128 f32 values into `s_v` and runs the seven butterflies at h=1,2,4,8,16,32,64. The implementation has one initial workgroup barrier and two per butterfly: **15 workgroup barriers** per gather CTA; each stage reads two shared values and writes one. The first five exchange partners never leave a 32-lane wave (the CTA's threads/lane indexing is contiguous), so a `shfl_xor` for h=1..16 has the same source-pair values. For each high-index element, keep the source subtraction in the original order **low minus high**, not `val - partner`; maintain the identical 32,64 LDS butterflies afterward. A proposed variant uses 5 shuffles, the initial LDS write, and the last 2 shared stages, with **5 workgroup barriers** total: ten fewer barriers and five fewer full-block LDS butterfly stages. This replaces memory/barrier traffic with lane-crossbar work; a speedup is plausible but unmeasured.

A scratch, non-device, asymmetric float32 exact-bit simulation exercised 256 scaled-int8 blocks, 100 mixed-magnitude random blocks and one all-zero block: **357 blocks × 128 outputs × 7 stages; PASS, all final f32 words identical** between the original shared-memory ordering and the five-wave-shuffle/two-LDS ordering. This is an arithmetic-and-lane-map proof, **not** a GPU bit-identity result; the compiled shuffle and compiler's exact FP contraction must still be checked. The scratch-only block rewrite has not been installed in the production tree or accepted as a speedup.

**Gate and disposition.** On gfx1100 compile all 17 embedding specialisations, check `llvm-readobj` for VGPR/SGPR/spill/LDS and resident waves against current native, verify actual BF16 output for all IQR embedding forms (valid IDs, negative/out-of-range IDs, run-boundary rows, scaling and rotated modes), run `r3_selftest embed_lookup_iqr`, and replay per-token cold/prefetch-warm with a queued GPU lease. A guard should follow resource records, not token count or a guessed latency threshold. **No code is enabled and no step-time improvement is claimed** while GPU access is reserved by a different session.

### Item 6 — activation requantisation once per block, not once per output tile

**Source trace.** The trunk IQR-GEMV (M 1–8) and IQR-M16 (M 9–16) interfaces already receive `X: const int8_t*` and its `XS: const float*` scales: the activation Hadamard/round-to-int8 work happened in the shared upstream activation-quantisation step, **not** inside the trellis weight decoder, per output-channel tile or per WMMA. M1's innermost loop simply loads a four-byte activation vector per lane and its token/K128 scale; M16's `bload` uses native 16-byte int8 fragments per token and K16 step, plus one precomputed `XS[token,kb]` fold. These bytes are reused by every weight-tile CTA through normal read-only cache; no second `rintf`, Hadamard, or scale selection is executed per output tile. The separate vocabulary head quantises its original BF16 activation in `vocab_quant_kernel` once per (token,128-K block), producing hi/lo int8 planes `XH` and `XL` plus `XS` for all head width classes and all output row tiles. The two passes are intentional for its 16-bit quantised activation and cannot collapse to a single 8-bit pass without changing the integer dot values.

**Gated-output exception accounted for.** The fused gate/up writer first creates the BF16 gated product in an output-plane scratch; all eight 16-column producers for the same 128-column group signal an arrival counter. Only the **last** CTA quantises each (token,128) group once (`gated_tail` runs one 128-wide Hadamard, one max reduction, one scale and one int8 pack for every group-token). Each CTA executes the tiny arrival protocol but the expensive requantisation is not repeated eight times. The open lever there is the global scratch/counter synchronization and full 128-element pass, not an unshared repeated quant; changing it alters a separate fused 17–32-row band explicitly excluded by this continuation.

**Opportunities and disposition.** The only within-kernel duplication observed is **loads of already quantised bytes/scales** by separate output-row tiles and by native WMMA half-waves. A per-block activation LDS staging variant would save repeated global loads at the expense of LDS capacity and extra barrier(s), and is not equivalent to hoisting quantisation; existing L2 residence means it might regress performance. Requantising again in the IQR tile would be redundant work and could change BF16 outputs by changing the rounding point. **No code change is warranted for item 6 on the traced hot paths** without a separate quantified activation-load bottleneck. No GPU replay or standalone bit-identity test has qualified such a staging alternative. The shipped single quantisation per token/K128 block is preserved.

### Report-only — trellis-routed MoE is a separate optimization family

The routed expert IQR entry point is `moe_gemm_q` via `r4d_moe_iqr.hip`, not the trunk `iqr_m16_kernel`. Its dispatch explicitly requires `kTrellisMad`, `kRowF32`, LRB0. It takes prequantised int8 activations and activation scales. A routed run of **one or two token rows** chooses `dot_run` with `sudot4` directly on lane-decoded four-byte columns, avoiding the WMMA transpose and fragments altogether. A run of three or more rows enters its own 16-row tile: decode each compressed K128 unit once, wave-LDS transpose, then eight shim WMMA steps per token tile; it looks ahead one K group and uses 128-K bias-seeded float folds.

The dispatch has a small decode/run grid and a larger mapped-run grid. The mapped grid reuses each weight decode across NT=4 token tiles, or NT=8 for a packed-run map. These are the other path's measured shape/occupancy tradeoffs: large NT makes register lifetime a bottleneck, even though decode amortisation improves. The source already documents that NT=16 spills and loses badly, while a dedicated decoder-plus-helper-waves approach was not better under measured idle-gap engine conditions. Because the MoE code lives in a separate unit, the 304/100 native-accumulation guard and any future trunk LDS permutation are **not inherited** by MoE. In MoE's longer-run path, native per-K128 accumulation could be analyzed separately, but requires its own resource, exact-byte, chain and routed-run gates over the varying expert occupancy and map modes. No MoE source was modified and no MoE speedup is claimed in this continuation.

### Current-native baseline build and extracted resources (before any experimental rewrite)

Built the merged native IQR path for gfx1100 with `nice -n 19`, Ninja serial execution (`-j1`), in an isolated scratch build directory, selecting the four IQR GEMM compile units and linking `libr3.so`. **Configure exit 0; compilation/link exit 0**, including all four IQR code-object partitions. The plugin reported the usual generated stubs and occupancy-target warnings; successful linking does not imply every specialisation meets the requested six-wave occupancy annotation.

A complete `llvm-readobj --notes` extraction of the four resulting gfx1100 code-object bundles yielded **3172 total kernels**, partitioned as **2176 GEMV**, **404 M16**, **544 M64**, and **48 miscellaneous**. The 404 M16 records have **zero VGPR spills and zero private-segment bytes**, with 115–228 VGPR per lane. Source-derived guard classification reproduces exactly **304 native and 100 compatibility M16** symbols. By token tiles: NT1 272 symbols / 260 native (115–155 VGPR; register-budget 9–13 waves); NT2 44 / 44 native (167–186 VGPR; 8–9 waves); NT3 44 / 0 native (183–216 VGPR; 7–8 waves); NT4 44 / 0 native (198–228 VGPR; 6–7 waves). These are **the current new-baseline numbers**, unlike the older resource tables used in earlier design sections.

The baseline alone has no changed symbols relative to itself; it is not a pass for an experimental native-extension guard. The build scope was **the four IQR GEMM units and the linked plugin**, not an unrestricted all-op full-plugin build. No selftests, bit-identity runs or cold/warm GPU replay were conducted here, because the queued GPU remains leased by another session. Any later candidate must be compared against all **3172** baseline records (including the 404 M16 and non-M16 records), then pass the full real-shape/chain/ragged identity and timing gates. The scratch code objects and full JSON notes are retained for exact comparison; neither an older release library nor the prior unrestricted candidate is being passed off as this baseline.

### Post-merge baseline provenance and item 3 prototype gate (2026-11-25)

**Moving-baseline correction.** The 3,172-symbol gfx1100 resource build above was configured from the branch at its earlier baseline (the merged 16-row native-fragment path, before a new origin/main merge). While this investigation's isolated prototype was compiling, a subsequent merge advanced the branch to include the independently developed **wide M17–64 native-accumulation change**. The earlier report's phrase “current new-baseline” therefore describes **the time of that build**, not the later merged HEAD. The 16-row 304-native/100-compat resource accounting below remains a like-for-like comparison of the exact same two pre-wide-merge source snapshots, but the entire 3,172-symbol table is **not** a freshly recompiled post-merge baseline. The later wide change is outside this continuation's ownership and was not modified or requalified here. A new candidate would require recapturing the merged baseline before claiming it is current.

**Completed scratch experiment for item 3.** Built and linked a four-unit IQR plugin containing an experimental long-token native path. For computed-trellis NT3/NT4 it streams the eight native weight fragments from the existing wave-local LDS transpose, reuses one activation B fragment plane instead of alternating two, and tests the native accumulate-once form. The four-unit candidate compiled and linked with exit 0. An asymmetric scratch-only CPU test of decoder-to-fragment ordering and the one-B-plane integer dots matched **10,240/10,240 int32 outputs** across NT1–4; this is not device byte identity and does not establish floating-fold or synchronization correctness.

Compared **all 3,172 compiled gfx1100 kernel records** against the matching pre-merge baseline by exact symbol name: **3,172 shared, 0 added/removed**. Of these **88 M16 variants changed**, exactly the 44 NT3 plus 44 NT4; **all 3,084 other records were resource-identical** (including NT1/NT2 M16). No new VGPR/SGPR spills, private bytes, or LDS usage. There were 44 VGPR improvements and 44 VGPR increases (delta **−68 to +46**). Critically, **34 M16 variants lose resident waves under floor(1536/VGPR)**, specifically **12 of 44 NT3** and **22 of 44 NT4**. The symbol-by-symbol scratch comparison and CSV are preserved. **Resource gate FAIL: reject this blanket extension** before any GPU bit or timing qualification. A resource-derived constexpr guard could *in principle* retain the improved specializations only, but such a partial variant must be recompiled against merged HEAD and pass full device byte identity and cold/warm no-regression gates before any code is accepted. The 12 exceptional NT1 stock forms were not included and remain on compatibility.

**Restoration and explicit exclusions.** The experimental M16 rewrite existed only during the isolated build. The worktree's native IQR source was restored to the newly merged HEAD, retaining the independent wide-row patch. No experimental kernel code was committed. GPU access was held by another queued long-running validation session throughout this checkpoint; **no IQR GPU selftest, chained/unchained M1–64 bit gate, ragged N=5376 replay, or six-shape cold/prefix-warm run occurred for this candidate**, and no release-performance conclusion is claimed. The embedding and vocabulary short-lookahead ideas likewise remain scratch designs only. This is a finished resource-rejection finding, not a shipping-ready optimization.

### Item 5 follow-up — compiled embedding wave-shuffle prototype and ISA/resource gate

Two isolated gfx1100 IQR-embedding-only plugins **built and linked successfully**, one baseline and one with five wave-local Hadamard butterflies using XOR cross-lane exchange, with identical source/compile settings except the scratch rewrite. The temporary rewrite was removed after compilation; no production source or patch is enabled.

Complete `llvm-readobj` comparison contains all **17** embedding kernel symbols, with exactly the same names, 17/17 resource rows and zero symbols added/removed. **Ten rotated specializations** (four table and six trellis B2–B7) produce changed ISA, while all seven nonrotated forms are unchanged. The rotated assembly has **15 → 5 workgroup barriers**, **0 → 5 cross-lane permutation instructions**, and **25–32 fewer static instructions per compiled specialization** (e.g., trellis B6 271 → 244). These counts come from actual compiled gfx1100 code objects, not a source-only estimate; they do **not** demonstrate timing gains.

The complete 17-symbol resource records show **no new spills, private bytes or LDS increases**. Sixteen records have identical VGPR counts; the rotated 4-bit table form increases **7 → 8 VGPR**, with unchanged 512-byte LDS. A conservative raw `floor(1536/VGPR)` test reports 219 → 192 potential waves for that record, but both counts exceed the architecture's finite wave-slot occupancy and cannot be interpreted as one actual lost resident wave. The precise capped occupancy needs hardware-resource validation before acceptance. In particular, it is incorrect either to declare a performance win merely from the lower static instruction count or to reject this low-register specialisation solely using an uncapped VGPR-floor formula.

The local float32 operand-permutation check passed 357 independent 128-element blocks with identical output bits, preserving low-lane minus high-lane subtraction. Nevertheless, **device BF16 identity, invalid-ID/boundary handling, the `embed_lookup_iqr` selftest and a short queued cold/warm per-token measurement have not run**: another GPU validation session holds the lease. Thus the candidate is *resource/ISA promising and correctness pending*, not approved for shipping. Its isolated compiled library and all notes/ISA counts remain in scratch for continuation. No live-per-step improvement percentage is claimed.

### Item 4 follow-up — compiled shorter-lookahead vocabulary head

Built and linked **both** current head-only gfx1100 plugin variants successfully using matching compile settings, with a scratch-only longer-term activation staging and weight-prefetch lifetime experiment. The changed version retains both activation planes and the existing integer accumulator/f32 fold; only for NT3/NT4, the next pair of K128 weight records is fetched after the K256 fold and block barrier instead of overlapping it with current decode/WMMA. NT1/NT2 preserve the existing lookahead. The scratch rewrite was restored on build completion, so no production source is enabled.

The exhaustive per-unit `llvm-readobj` resource comparison found **six of six matching kernels** (four `vocab_tile_kernel` specializations plus two support kernels), exactly **two changed**. Both changed symbols keep their VGPR counts and register-budget wave floor, LDS, and SGPR allocation; neither introduces a spill or larger private allocation. Measured compiler records:

| Head tile | Baseline VGPR | Experiment VGPR | VGPR spills before → after | Private segment bytes before → after | LDS bytes | Register wave floor |
|---|---:|---:|---:|---:|---:|---:|
| NT3 direct | 240 | 240 | **35 → 9** | **52 → 28** | 42752 | 6 → 6 |
| NT4 direct | 256 | 256 | **797 → 663** | **204 → 204** | 51200 | 6 → 6 |

This passes the **measured resource no-regression test for all six head symbols** and substantially reduces NT3 spill instructions. Nonetheless NT4 still has 663 spills and 204 bytes private; more fundamentally the eliminated cross-unit prefetch could serialize global loads and slow actual token time even with fewer spill instructions. No speed claim follows from lower spills alone. The source-level ordering preserves K128 indices, integer accumulation and f32 scale folds, but this is **not** an executed GPU BF16 identity test. The full `logits_gemm_vocab_iqr` selftest and the real M33–64 chained/unchained head cold+warm replay are outstanding because another session holds the GPU lease. The variant is therefore **resource-qualified only, not shipping-qualified**. The published native head remains unchanged; evidence and both isolated plugin builds are kept in scratch.
