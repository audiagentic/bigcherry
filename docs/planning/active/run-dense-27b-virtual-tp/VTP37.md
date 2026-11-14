---
id: VTP37
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-12T23:29:57.750388+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P0
---

# int8 (w8a8) TP3 decode and prefill: pipelined prefill on rotated-int8 trunks, gfx1100 int8 row selection, three-rank fused AR + norm + int8 quant

## Description

**First priority (owner direction 2026-11-13): raise int8 (w8a8) TP3 decode and prefill.** World: dense Qwen3.8-27B w8a8 (`qwen3.8-27b-w8a8-df2.rad`, plugin `qwen35_w8a8` on the shared `qwen35_fp8` trunk) on XTX + XTX (gfx1100, native int8, emulated fp8) + R9700 (gfx1201), host-staged exact all-reduce, layout `qwen27b-tp3-int8` (kv 1,1,2; ffn 6656/6656/4096; lk 6,6,4; lv 18,18,12; tune_pin all_reduce:nblocks=4,nthreads=256; DFlash2 depth 3; fp8 KV).

**Baseline (2026-11-12, VTP14 N6 refresh, build 8a6f2776, 2x ABBA):**

| | 1 / 4 / 6 / 8 streams tok/s | one-prompt prefill tok/s 512 / 2048 / 8192 |
|---|---|---|
| w8a8 TP2 (two XTX) | 65.3 / 175.9 / 167.9 / 194.4 | 1438 / 1487 / 1903 |
| **w8a8 TP3** | **83.6 / 188.0 / 179.2 / 213.1** | **1547 / 1615 / 2181** |
| MXFP4 TP3 (reference) | 115.6 / 222.7 / 214.2 / 239.3 | 1848 / 1919 / 2709 |

**Why int8 trails MXFP4 TP3 (assumptions to validate, each with its source):**

1. **No pipelined prefill on int8.** `arch/qwen35_fp8/qwen35_fp8.cpp` (~line 1074) leaves `m.pipe_on` false when `m.a_h.q8_fed` (the w8a8 trunk's norms write rotated int8 codes via `rmsnorm_had_quant_i8g`), printing 'the pipelined prefill stays off on this trunk'. So rad0020, rad0025, rad0026 and rad0027 never run on int8. On MXFP4 they were worth: one-prompt TTFT -15 to -19% (rad0020), four-prompt TTFT 10.5 -> 9.0 s and 8 streams 214 -> 240 tok/s (+12%, rad0025+rad0026), a long prompt beside generating streams -19% (rad0027). The stated reason is that `NormQuantFP8::step_plain` (arch/common/rad_fp8.h ~486), used by the pipeline's boundary norms on lane 1, writes `a.q`/`a.s` (E4M3) while `step()` writes `hi8 ? a.q8 : a.q` / `hi8 ? a.s8 : a.s`. `op` is already declared as `rmsnorm_had_quant_i8g` when `hi8`. Hypothesis: the fix is to make `step_plain` write the int8 pair when `hi8`, add `a.q8`/`a.s8` of the pipelined activations to the lane-1 concurrent set (`rad_buf_concurrent`), check every other lane-1 producer (GateQuantFP8 / `gated_had_quant_i8g`, `a_go`, `a_attn`, `a_ffn` q8_fed paths) and drafter taps, then drop the q8_fed refusal.
   - Corollary: VTP31 experiment 12's int8 'pipeline floor 256' gain (+0.3-1.3% decode, -1.7/-2.2% TTFT) cannot be the pipeline (it never ran); treat it as noise and fix the comment on `tuning.pipelined_prefill_min_rows` in the `qwen27b-tp3-int8` layout.
2. **gfx1100 int8 GEMM row selection at M 9-128.** RAD47 fixed this band for MXFP4 on gfx1100 (tiled row 4-17x faster than the skinny/decode rows from M=17; +27% / +37% at 4 / 6 streams on TP2). The int8 rows (`gemm_i8a8_nt_m16`, `gemm_i8a8_tiled`, `gemm_i8a8_rb8_tiled`, `gemm_i8a8_rb32_tiled`) were not swept the same way. RAD48 recorded an open M 33-64 crossover: `gemm_i8a8_nt_m16` 358 us at M=32 and 772 us at M=64 against the native tiled row ~550 us flat (N=34816 K=5120, one XTX). Multi-stream decode with 3 drafts runs M = 4-32 rows per step; the last prefill chunk and mixed steps hit 33-128. Per-rank N is smaller in TP3 (ffn 6656 per XTX rank), so the crossover must be re-measured at TP3 shapes.
3. **No three-rank fused all-reduce + RMSNorm + Hadamard int8 quant.** `NormQuantFP8::declare` declares `ar_rmsnorm_had_quant_i8g` only when `g.world == 2`; TP3 runs `all_reduce` then `rmsnorm_had_quant_i8g` (a separate full-row pass that also does the Hadamard rotation). GPT's estimate for the fp8 twin is under 1% of prefill; the int8 pass is heavier (rotation), so measure its share before building.
4. **int8 kernel efficiency.** RAD48's native gfx11 IU8 tile (on main, #601): one XTX M=2048 N=34816 K=5120 25.8 -> 8.33 ms (3.1x, ~88 TFLOP/s); end-to-end int8 prefill and KLD never recorded. XDP08-A proposes per-row weight-scale LDS staging for `gemm_i8a8_tiled` (only weight scales are unstaged; upstream reports +4% from staging activation scales). Decode GEMV (M 1-8) bandwidth on both archs is unmeasured; int8 weights are ~2x MXFP4 bytes, so 1-stream decode is expected to trail MXFP4 - compute achieved GB/s per rank to see the headroom.
5. **Split.** The int8 split was swept (VTP14 N6); keep it unless the pipeline changes the balance (re-check 6,6,4 vs neighbours after step 1).

## Steps

1. **Validate the assumptions above** from the source and a short profile (`--profile-ops`, int8 TP3, one 2048-token prompt and 8-stream decode): confirm pipe_on is false today, the per-op share of `all_reduce` + `rmsnorm_had_quant_i8g`, the int8 GEMM rows selected at each M per rank, and achieved decode GB/s.
2. **Pipelined prefill for rotated-int8 trunks** as a managed patch (next free number after rad0029, plan-ids VTP37): `step_plain` writes the int8 pair when `hi8`; lane-1 buffer sets include the q8/s8 activations; remove the q8_fed refusal (keep a refusal for any producer that still cannot write int8 on lane 1). Card-free test applying to the pinned source. Build; smoke; greedy identity serial vs pipelined on int8 TP3 (`--pipelined-prefill off` vs `auto`); int8 stress (12 runs, rad0028 guard); 2x ABBA vs the current int8 layout: TTFT 512/2048/8192, four 4096-token prompts, 8192 (or 6144 at max-model-len 8192) beside 4 generating streams, 1/4/6/8 streams.
3. **gfx1100 int8 row crossover:** rad-kbench all-rows on the int8 rows at the TP3 per-rank shapes (N, K from the int8 split) for M 1-256; if a faster valid row loses on priority, change libr3's gfx11 constraints/priority (like RAD47, gfx1201 untouched); ABBA end to end; KLD vs BF16.
4. **Three-rank fused `ar_rmsnorm_had_quant_i8g`** (libr4d `r4d_fused_quant_fp8.hip` / `rad_ar_rmsnorm_quant`, `ar_fused_ok`), only if step 1 shows the separate pass is >= 3% of int8 prefill or decode device time.
5. **int8 tile weight-scale LDS staging** (XDP08-A) and decode GEMV, only if step 1 shows the kernel time dominates after steps 2-3.
6. Fold winners into `qwen27b-tp3-int8`; refresh the baseline with 3x ABBA against int8 TP2 and MXFP4 TP3.

3a. **gfx1100 int8 rows at 17-32 rows (2026-11-14).** `kbench-1/xtx.md`: `gemm_i8a8_nt_m16` is flat to M=9 (12288 x 5120: 91 us, ~691 GB/s) and costs 162 us at M=18 and 144 us at M=32; the R9700's row is flat to M=32 (105-108 us). Six and eight streams at depth 3 issue 24 and 32 rows, and the XTX ranks set the decode pace. Sweep every M from 10 to 32 at the TP3 per-rank shapes; if the band holds, a row that reads the weight once for up to 32 rows in libr3, gfx1201 untouched. ABBA at 4/6/8 streams with decode-only tok/s; KLD against BF16; stress. Expected decode-only +9% to +22% at 6-8 streams.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Greedy identity: pipelined vs serial on int8 TP3 (same build, same split).
- int8 stress: 12 runs of the rad0028 guard (4/8 streams + 6144-token prompt beside 4 generating streams), zero faults.
- KLD against BF16 for any kernel-row change (rows of a different accumulation order).
- Matched ABBA (2x per lever, 3x for the final refresh): 1/4/6/8 streams with speculation, one-prompt TTFT 512/2048/8192, four 4096-token prompts, long prompt beside 4 streams.

## Effort & Risk



## Standards



## Acceptance Criteria

- int8 TP3 runs the pipelined prefill (startup receipt shows pipe_on) with no stress faults, and passes the **reordering quality gate** below.
- **Reordering quality gate** (owner decision 2026-11-13; for any change that keeps the arithmetic exact but changes its order or partition: pipelining, chunking, collective method, split). Bit identity to serial is not required. Both checks must pass, on the same KLD corpus (42,761 positions):
  1. *Noise floor*, against a serial reference of the same build and layout: KLD mean, KLD p99.9 and top-1 no worse than a serial run that changes only the prefill chunk size (2048 -> 1024 tokens). A serial-vs-serial repeat must show the reference deterministic. int8 TP3 floor (2026-11-13): mean 0.00197, p99.9 0.059, top-1 97.81%. rad0030: 0.00125 / 0.021 / 99.31% - pass.
  2. *No quality loss against BF16*: KLD mean and p99.9 within +10%, top-1 within -0.2 points and PPL within +0.1% of the format's recorded values. int8 TP3 (VTP14 N6): mean 0.00199, p99.9 0.043, top-1 97.32%, PPL +0.03%. rad0030: not yet run (held 2026-11-13).
  MXFP4 uses the same rule with its own floor (to be measured once) and its recorded BF16 values (VTP14: mean 0.0435, p99.9 1.16, top-1 87.07%, PPL +2.64%).
- int8 TP3 one-prompt TTFT improves by at least 10% at 2048 tokens and 4/6/8-stream tok/s by at least 5% over the 2026-11-12 baseline, or each lever's NO-GO is recorded with its measurement.
- Every change is a managed patch or layout setting (no model or card constants), selectable per run.

## Notes

Consolidated 2026-11-13. Carries RAD48's open work (superseded into this item) and XDP08-A (int8 weight-scale staging). Related: VTP36 (TP3 performance generally, MXFP4 2048-token gap), RAD47 (gfx1100 MXFP4 row selection, implemented), VTP32 (split solver). Patched source used for the assumptions: the current build's source tree (rad0016-rad0029 applied). Implementation and validation of the assumptions: handed to a Fable 5.2 agent (2026-11-13).

**2026-11-14.** The M 33-64 NO-GO stands (no arm reaches it). The M 17-32 band was not assessed: nt_m16 beats the tiled row there, but against its own 1-9-row time it is 1.6-1.8x, which matches the 6-stream dip present in every table since VTP14 (`docs/evidence/2026-11-14-tp3-long-context/README.md`).

## Results (2026-11-13)

Builds: base `9a5d8194a35fcffae6b8b43c2450d586` (main, rad0016-rad0029); with rad0030
`ccdd65fa6acbbc3057a15b530827dcde`. Logs: `${BC_TASK_TMP}/vtp37/logs/` (directory names below). World: the
`qwen27b-tp3-int8` layout's settings (exact wire, DFlash2 depth 3, fp8 KV, max-model-len 8192, 8 sequences, 2048-token
chunks, 256-row floor, all-reduce 4x256), every arm a fresh start under the gpu lease.

### Assumptions

| # | assumption | result | evidence |
|---|---|---|---|
| 1 | no pipelined prefill on int8; the blocker is `step_plain` writing the E4M3 pair | **held** | startup note on the base build (`base-smoke`); `step_pipelined` read end to end; with the write fixed the pipeline runs with no other change needed |
| 1a | the lane-1 set needs the q8/s8 pair | **held** | `rad_buf_concurrent` named `a_h.q` / `a_h.s`; now the fed pair |
| 1b | other lane-1 producers need work (`gated_had_quant_i8g`, `a_go`, `a_attn`, `a_ffn`, drafter taps) | **failed** | none of them runs on lane 1: `pipe_boundary` issues only the all-reduce, the plain norm, the add and the final norm (bf16 planes); the gated quantiser, attention gate and delta-net output quantiser are issued by the blocks' piece paths on lane 0 and already read `cq()` / `cs()`; `pipe_taps` is on lane 0 |
| 1c | VTP31 experiment 12's int8 floor gain was noise | **held** | the base build prints "stays off on this trunk" with the 256-row floor set, so the floor had no effect |
| 2 | gfx1100 int8 rows at M 9-128 pick slow rows, as MXFP4 did | **failed for M <= 32 and M >= 65, held weakly for M 33-64** | `kbench-1`: `gemm_i8a8_nt_m16` (selected to M 64) is 1.4-2.4x faster than the tiled row to M 32 and 1.07-1.9x slower at M 33-64; from 65 only the tiled row serves. Not RAD47's 4-17x. The R9700's selection is right at every M |
| 3 | the separate `rmsnorm_had_quant_i8g` pass is worth fusing at three ranks | **failed** | `prof-base`: 0.9-1.1% of prefill device time, 1.4-1.6% at 8 streams, 2.1-2.3% at 1 stream; under the 3% gate, and a fused form saves the launch, not the pass |
| 4 | int8 kernel time dominates | **failed** | `prof-base`: the all-reduce is 52% of prefill device time on the XTX ranks and 81% on the R9700 (40% / 73% at 8 streams); every int8 GEMM together is 36% / 12% of prefill. Isolated decode GEMV (`kbench-1`, M 1-9, cold weights): 690-715 GB/s on the XTX, 570-600 GB/s on the R9700, so little bandwidth headroom |
| 5 | keep the 6,6,4 split unless the pipeline changes the balance | **open** | `split-1`: with the pipeline on, 5,5,6 measures +1.3 / +0.1 / +4.0 / +2.9% at 1/4/6/8 streams, TTFT +3.2 / -0.2 / -0.3% at 512/2048/6144 (2 arms each); not changed here |

Profile, base build (`prof-base`, `--profile-ops`, device ms and share, XTX / XTX / R9700):

| workload | device ms | all_reduce | int8 GEMMs (gate_up, down, ssm_inz, ssm_out, attention) | gdn_chunk_scan | rmsnorm_had_quant_i8g | drafter |
|---|---|---|---|---|---|---|
| one 2048-token prompt | 3959 / 3958 / 3947 | 52.3 / 52.6 / 81.4% | 36 / 36 / 12% | 4.9 / 4.9 / 0.7% | 0.9 / 0.9 / 1.1% | 1.6 / 1.5 / 0.8% |
| 8 streams | 6429 / 6430 / 6393 | 40.2 / 40.3 / 72.9% | 41 / 41 / 16% | 2.4 / 2.4 / 0.3% | 1.4 / 1.4 / 1.6% | 5.0 / 5.1 / 1.5% |
| 1 stream | 3269 / 3268 / 3260 | 29.8 / 30.0 / 60.4% | 43 / 43 / 20% | 2.7 / 2.7 / 0.4% | 2.1 / 2.3 / 2.2% | 5.6 / 5.6 / 3.2% |

The R9700's all-reduce share is mostly waiting for the XTX ranks. With the pipeline on (`prof-on`) the XTX ranks'
GEMMs cost more in total (two half-width pieces) and the R9700's all-reduce share of a 2048-token prompt falls from 81%
to 65%.

int8 rows on one XTX (`kbench-1/xtx.md`, `rad-kbench --all-rows --timing-only`, the 27B w8a8 fixture's shapes, us;
selected row first):

| N x K | M 1-9 nt_m16 / tiled | M 18 | M 32 | M 33 | M 64 | M 65-128 tiled |
|---|---|---|---|---|---|---|
| 5120 x 6144 | 73 / 167 | 133 / 169 | 103 / 167 | 232 / 167 | 302 / 168 | 168-170 |
| 12288 x 5120 | 91 / 216 | 162 / 215 | 144 / 215 | 232 / 215 | 308 / 215 | 218-221 |
| 16384 x 5120 | 126 / 297 | 212 / 293 | 198 / 297 | 344 / 297 | 526 / 295 | 292-294 |
| 34816 x 5120 | 250 / 600 | 413 / 602 | 379 / 601 | 626 / 590 | 831 / 603 | 608-621 |

The fixture is the one-rank model's; the TP3 per-rank shapes (13312 x 5120, 5120 x 6656, 6144 x 5120 on an XTX rank) lie
between its rows.

### Ranked assessment

1. **Pipelined prefill on the int8 trunk (done, rad0030).** The all-reduce is the largest cost at every workload and
   nothing overlapped it on int8.
2. **The all-reduce itself** (VTP36's shared work: fewer bytes, a reduce centred on the XTX ranks). Still 43-57% of XTX
   device time and 63-65% of the R9700's after rad0030. This is where the remaining gap to MXFP4 is.
3. **Split 5,5,6 against 6,6,4** with the pipeline on: +3-4% at 6/8 streams in two arms. A layout setting; needs a 3x ABBA.
4. **gfx1100 int8 row at M 33-64** (NO-GO for now): the tiled row is up to 1.9x faster there, but the layout never issues
   that band in decode (8 sequences x 4 rows = 32) and a prefill chunk lands in it only as a 33-64-row tail. No arm of the
   acceptance workload reaches it, and the offline KLD runs 2048-row chunks, so neither a gain nor quality could be
   shown. It matters at more than 8 sequences. The change would be a second, lower-priority `gemm_i8a8_nt_m16` row for
   M 33-64 in libr3 (a cap alone would leave an N that is not a multiple of 128 unserved).
5. **Three-rank fused `ar_rmsnorm_had_quant_i8g`** (NO-GO): 1-2% of device time.
6. **int8 tile weight-scale staging and decode GEMV** (NO-GO): the kernels are not the dominant cost and decode GEMV is
   at 72-94% of the cards' nominal bandwidth in isolation.

### rad0030: 2x ABBA against the base build (logs `abba-1`)

Order A B B A A B B A, A the base build, B rad0030 with `--pipelined-prefill on`. Mean ± standard deviation.

| | base | rad0030, on | change |
|---|---|---|---|
| 1 stream, tok/s (8 rounds) | 83.1 ± 0.2 | 88.1 ± 0.8 | +6.1% |
| 4 streams | 187.4 ± 4.6 | 201.2 ± 3.7 | +7.4% |
| 6 streams | 179.6 ± 2.2 | 193.7 ± 3.3 | +7.8% |
| 8 streams | 214.3 ± 4.1 | 232.3 ± 1.4 | +8.4% |
| one prompt, 522 tokens, time to first token (8) | 342 ± 7 ms | 281 ± 10 ms | -18.0% |
| one prompt, 2021 tokens (8) | 1252 ± 14 ms | 1040 ± 15 ms | -17.0% |
| one prompt, 6059 tokens (8) | 3757 ± 25 ms | 2959 ± 28 ms | -21.3% |
| four 4096-token prompts, slowest time to first token (4) | 10121 ± 78 ms | 8492 ± 46 ms | -16.1% |
| 6144-token prompt beside 4 generating streams (4) | 4049 ± 37 ms | 3380 ± 24 ms | -16.5% |

One-prompt prefill is now 1860 / 1944 / 2048 tok/s at 522 / 2021 / 6059 tokens (base 1526 / 1614 / 1613). Every server
was alive at the end of its arm. The decode rows gain because each round includes its 512-token prefills (pipelined
from 256 rows) and mixed steps.

### Output: pipelined against serial

**Not bit-identical at every prompt length; within the serial step's own sensitivity to chunk size.**

- Greedy text (`ident-1`, one stream, 96 new tokens, serial against pipelined on the rad0030 build, two arms each):
  identical at 300, 508, 4016, 6027 and 6090 tokens (the last beside 4 streams); different at 1092, 2023 and 2955
  tokens (the texts part at characters 157 and 212, and past character 400), at a near tie. Serial against serial, serial against the base build, and pipelined
  against pipelined are identical at all seven. The 4-stream cases differ between two runs of one configuration, as on
  the base build.
- First-token top-20 log-probabilities (`lp-1`): serial against serial bit-identical at 8 prompt lengths; serial
  against pipelined bit-identical under 1024 tokens, and from 1116 tokens the top token is the same with its
  log-probability within 0.001 (largest difference among the 20: 0.62, in the tail). MXFP4 TP3, where the pipeline has
  been on since rad0020, differs from its serial step in the same way (largest 1.53).
- The pipelined step is the serial step over two pieces: against a serial run at 1024-token chunks the pipelined
  log-probabilities are bit-identical at 2072 and 4003 tokens, where the piece bounds coincide with that run's chunks.
- int8 TP2 (`tp2-1`): `on` starts and pipelines; first-token log-probabilities bit-identical to serial at 7 of 8 lengths.
- KLD against the serial prefill (`kld-1`, 24 documents, 42,761 positions, top-64 reference recorded serial):

| scored against serial, 2048-token chunks | KLD mean | median | p99 | p99.9 | top-1 | PPL ratio |
|---|---|---|---|---|---|---|
| serial again | 5e-11 | 0 | 1e-8 | 2e-8 | 100% | 1.0000 |
| pipelined (`on`) | 0.00125 | 2e-9 | 0.0038 | 0.021 | 99.31% | 0.9992 |
| serial, 1024-token chunks | 0.00197 | 0.00036 | 0.0078 | 0.059 | 97.81% | 1.0004 |

For scale, w8a8 against BF16 is 0.0020 mean (VTP14 N6). The upstream comment that a pipelined step is "bit for bit the
serial one" does not hold on this world, for either trunk.

### Stress (logs `stress-1`)

rad0028's guard pattern on the rad0030 build with `--pipelined-prefill on`: a 4/8-stream bench of 512-token prompts,
then 4 streams generating 1500 tokens with a 6144-token prompt beside them.

| build | pipelined prefill | runs | faulted | benches failed | mean 4 / 8 streams | long prompt's text |
|---|---|---|---|---|---|---|
| ccdd65fa (rad0030) | on | 12 | 0 | 0 | 210.7 / 241.9 tok/s | the same in all 12 |

### Decisions taken

- rad0030 is opt-in: `--pipelined-prefill on`. `auto` keeps a rotated-int8 trunk serial, so no default changes.
- `qwen27b-tp3-int8` sets `tuning.pipelined_prefill = "on"` (owner decision 2026-11-13), and its comment on the 256-row
  floor no longer claims VTP31 experiment 12's gain.
- Acceptance: 2048-token TTFT -17% (>= 10%) and 4/6/8 streams +7.0 / +8.1 / +9.0% over the 2026-11-12 baseline (>= 5%).
  "Greedy output identical to serial" is not met as written; the measured difference is above.

### Smoke and power settings

- Launcher smoke (`layout-smoke`): `launch-flashnext-best.sh start --layout qwen27b-tp3-int8` on the rad0030 build shows
  `--pipelined-prefill on [manifest]` in the startup receipt, prints the "are pipelined" note on the first wide step and
  answers the greedy canary.
- The launcher applies the cards' configured power settings (XTX: -75 mV, sclk 2900 MHz) and restores them only in its
  own `stop`. The first smoke of this item was stopped by PID, so those settings stayed applied for every measurement
  above, both arms alike: the tables compare like with like but are not at the driver defaults the 2026-11-12 baseline
  names. The settings were restored from the journal afterwards (`gpu-power restore`).

### Not done

- Step 3 as a change (the M 33-64 row), steps 4-5 (NO-GO above), step 6's 3x refresh against int8 TP2 and MXFP4 TP3.
- KLD against BF16 with the pipeline on (the serial-reference KLD above stands in for it).
- An 8192-token one-prompt arm (the layout's max-model-len is 8192, so 6144 was used).

## Change Log

- 2026-11-12T23:29:57.750388+00:00 (created-by): Created by agent
- 2026-11-12T23:30:35.740511+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
- 2026-11-14T06:59:30.762472+00:00 (updated-by): Updated: section:steps, section:notes
