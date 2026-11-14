# qwen27b-tp3-int8-556 (and -long / -long-bf16): where prefill and decode time goes, and what to do about it

2026-11-14. Analysis (agent review; the libr3 claims were spot-checked: the libr3 `O_PSPLIT`
static_assert, the gfx1100 `gemm_i8a8_nt_m16` M=18/32 kbench rows and `--prefill-span`). Nothing was run on the GPUs for
this analysis. Inputs: plan items (VTP36, VTP37, VTP38, VTP32, HL02, RAD38, RAD50, RSC12, PWR06, ENG01-04), engine source
the engine source of the 2026-11-14 build, libr3 (`engines/radiance/components/libr3`), the profiles
and A/B arms under the run folder (`ab/`), the full-cache run, and VTP37's kbench and profile logs
(VTP37 logs).

**Read this first.**
- Every A/B arm is one fresh start (no repeats). VTP36 measured 7% between starts for a 1024-row all-reduce. Nothing
  below is a config change until it has repeats / ABBA.
- `--profile-ops` numbers are summed device events with restored synchronisations: used for shares and per-call ratios.
- "Measured" = read from a log. "Derived" = arithmetic on measured values. "Inferred" = a reading with a confirming
  measurement.

## 1. Measurements

### 1.1 A/B arms (served `-long-bf16` argv, one start each; `ab/<arm>.json`, `ab.log`, `ab2.log`)

| arm | TTFT 100,366 tok | TTFT 8,003 tok | decode 1 / 8 streams chunks/s | interference: long TTFT | decode chunks/s during it (4 streams) | gap p50 / max |
|---|---|---|---|---|---|---|
| base (2048) | 53.78 s (1866 t/s) | 2.915 s | 33.7 / 87.8 | 61.7 s | 4.96 | 934 / 1549 ms |
| `--max-num-batched-tokens 4096` | **44.68 s** (-17%) | 2.915 s | 33.7 / 88.7 | 51.3 s | 4.11 | 148 / **2441** ms |
| `--prefill-interleave 2` | 53.84 s | 2.894 s | 33.7 / 88.0 | 66.8 s (+8%) | **13.5** (2.7x) | 41 / 1547 ms |
| `--busy-prefill-chunk 512` | 53.85 s | 2.914 s | 33.7 / 88.1 | 89.9 s (+46%) | 8.96 | 451 / **677** ms |
| 4096 + interleave 2 | 44.69 s | 2.907 s | 33.7 / 90.1 | 54.4 s (-12%) | 11.1 (2.2x) | 40 / 2412 ms |
| `--tp-attn-split rows` | refused: "the 'qwen35_w8a8' plugin does not split attention by rows for this model" | | | | | |

### 1.2 Per-op profile, one 100,366-token prompt (`ab/prof-prefill.log`)

| op | XTX rank 0 ms (share), us/call | XTX rank 1 | R9700 rank 2 |
|---|---|---|---|
| all_reduce (12,733 calls) | 36,082 (41.9%), 2834; lane idle before it 18,980 ms | 36,386 (41.9%) | 47,777 (55.6%), 3752; idle 7,068 ms |
| attn_paged (16 layers x 99, ~1024 rows) | 21,468 (24.9%), **13,510** | 21,295 (24.5%) | 14,295 (16.6%), **8,996** |
| gemm_nt_q | 20,298 (23.6%) | 20,412 | 19,732 (23.0%) |
| gdn_chunk_scan (48 x 99) | 5,789 (6.7%), **1218** | 6,374, 1341 | 664 (0.8%), **140** |

Decode profile (`ab/prof-decode.log`): XTX gemm_nt_q 45.9%, all_reduce 30.9%, prefetch 6.5%, attn 2.1%; R9700
all_reduce 40.3%, gemm 35.4%, prefetch 10.6%.

### 1.3 Two-term TTFT fit (derived): TTFT(N) = a N + b N^2

| | a (s/token) | linear at 100k | b (s/token^2) | quadratic at 100k |
|---|---|---|---|---|
| 2048-token steps | 3.494e-4 | 35.1 s | 1.858e-9 | **18.7 s** |
| 4096-token steps | 3.572e-4 | 35.9 s | 0.877e-9 (**x0.47**) | 8.8 s |

Predicts 27.65 s at 60k (measured 27.2 s bf16). 4096 steps leave the linear term and halve the quadratic.

## 2. Reading

a. **Short context = the all-reduce chain** (4 x 2.83 ms a layer x 64 = 0.72 s per 2048 tokens; 12,544 calls x 2.83 ms
   = 35.5 s against the 35.1 s linear term). VTP38 / VTP36 / HL02 territory.
b. **At depth the extra time is XTX attention on the critical path**: quadratic term 18.7 s, XTX attn 21.5 s profiled,
   XTX all-reduce lane idle 19.0 s.
c. **XTX attention launch underfills the card at 1024-row pieces.** An XTX rank holds one KV head (`kv=1,1,2`): a
   1024-row piece is 16 768-thread workgroups on 48 WGPs. ~23-26 TFLOP/s against a 142 TFLOP/s f16 roof; the R9700
   (2 KV heads, 32 WGPs) reaches ~70. Not memory (~60 GB/s). The kernel is the proper tiled flash prefill
   (`r4d_attn_prefill_h256_gqa6`), not a decode kernel.
d. **The KV split meant to fix this is not taken, or runs in emulated form**: libr3's native instantiation
   `static_assert(!R3N || !(OPT & O_PSPLIT))` (`native/r4d_attn_prefill_h256_gqa6.hip.rw:28`); lab history: compat form
   369 us vs native 204 us. Other candidates: `wgp_count`/scratch on the pipelined piece path; a one-workgroup-a-WGP
   law that is too conservative. One kbench matrix decides.
e. **After the all-reduce, XTX compute is next**: ~8.9 ms a layer-step against the chain's 11.3 ms (~80%). XTX GEMM
   82 TOPS at M~1024 (57% of the iu8 roof); delta-net scan ~10x slower per head than the R9700 (compat form; the native
   form was wrong, KLD 5.52, shelved).
f. **Decode is XTX weight reads.** Isolated `gemm_i8a8_nt_m16` 12288x5120: 91 us at M 1-9 (691 GB/s), **162 us at
   M=18, 144 us at M=32**; the R9700's row is flat to M=32 (105-108 us). 6 / 8 streams at depth 3 = 24 / 32 rows.
g. **The "8 streams 281 tok/s" headline is ~40% prompt phase**; decode-only ~570-640 tok/s.

## 3. Ranked opportunities

1. **RAD52 (new, P1, M): prefill attention fills the card at every piece size.** Expected TTFT at 100k -10% to -22% at
   2048-token steps; long prefill beside decoders with small chunks -20% to -35%; nothing <= 8k or in decode. Confirm
   card-level first: selftest/kbench `attn_prefill_h256_gqa6`, kv_heads 1/2, rows 128-4096, context 8k/50k/100k, forced
   splits 1-12, native vs compat. Kill: cost per row already flat in isolation (then trace the engine launch).
2. **VTP36 step (P1, S, config only): wide steps only for a lone prefill** via `--prefill-span`:
   `checkpoint_interval 2048` + `max_num_batched_tokens 4096` + `prefill_span 2` (alone 4096, shared 2048). Predicted
   ~44.7 s at 100k with base-like interference (gap max ~1.55 s). Arms also with interleave 2, and 8192/span 2.
3. **VTP39 (new, P1, M): long prompts beside other work.** Interleave trade is linear (N=2 +8% for 2.7x decode; N=8
   ~+25%). Full-cache: eight 100k prompts mean TTFT 445 s round-robin vs ~245 s one at a time; recompute <= 7% of
   steps; short requests waited ~330 s for a slot. Levers: decode share by time, a cap on concurrently prefilling
   sequences, short-request priority (a prefilling runner is never a victim, so priority alone cannot free a slot).
4. **VTP37 step 3a (P2, M): gfx1100 int8 row at 17-32 rows** — decode-only +9% to +22% at 6-8 streams; explains the
   long-standing 6-stream dip. Confirm with a kbench M 10-32 sweep at TP3 per-rank shapes.
5. **RAD53 (new, P2, S): gfx1100 fp8 KV read** — software E4M3 conversion (`r3_compat.h` `r3_e4m3_to_f32`, ~15 VALU
   per element) on every K/V read; fp8 60k decode step 39.4 vs 32.6 ms bf16. E4M3 is exact in f16: direct shift/add
   conversion, bit-identical. Expected fp8 at depth to bf16 parity. Confirm per-rank with one profile per dtype.
6. **VTP40 (new, P2, S): speculation window by load** — at one stream rows 5-8 are nearly free; contingent on whether
   DFlash2 accepts other depths. Read per-position acceptance from `/metrics` first.
7. **VTP38 sizing (notes):** Gate B's -17% on the call = ~-15% TTFT at 2048 tokens, ~-11% at 100k, nothing in decode;
   ceiling ~-20% because XTX compute is at 80% of the chain. Adds to RAD52 (different terms): 53.8 -> ~44.7 -> ~38-39 s.
8. **VTP32 (note, P3):** R9700 waits 69 us of 370 us decode cycle at 5,5,6; ~4.5/4.5/7 balances (+9% at 8 streams,
   ~-9% cache). Solve after VTP37 3a.
9. **Unexplained:** R9700's first GEMM after each all-reduce costs ~0.65 ms extra in pipelined prefill only (VTP37
   `prof-base` vs `prof-on`). Off the critical path today. Diagnostic arms for HL02 and PWR06 (gfxoff kernel).

## 4. Ruled out

- Peer-to-peer / slot moves: no p2p on this host; moving the R9700 keeps some card on the x4 link.
- `--tp-attn-split rows`: refused by the plugin; would also need every head's KV on every rank.
- More KV heads on the R9700, FFN off the x4 rank: cost cache capacity / decode pace.
- 8192-token steps: predicted 42-43 s vs 44.7 s at 4096.
- `--busy-prefill-chunk 512` as is: +46% long TTFT; revisit after RAD52.
- Preemption recompute: <= 7% of full-cache steps.
- Decode GEMV at 1-9 rows, XTX prefill int8 tile: at isolated rates.
- Host per-step cost / powersave governor: 8k TTFT identical at 2 and 4 steps, decode p50 identical across five
  starts; one governor arm is cheap, expected ~0.
- Weight prefetch: device time beside the collective; one `--weight-prefetch off` arm would confirm (expect within 2%).

## 5. Suggested order

1. One lease window, card-level: RAD52 attention matrix and VTP37 3a row sweep (kbench only).
2. One window, engine: `--profile-ops` at 4096 steps (100k); fp8 vs bf16 decode profile at 60k; `/metrics` acceptance;
   VTP36 lone-prefill arms with repeats.
3. Build RAD52 and RAD53 (libr3 only); VTP39 step 1 settings in parallel.
4. VTP38 continues unchanged; read its acceptance with the XTX compute ceiling.

The full item texts (RAD52, RAD53, VTP39, VTP40) and the add-text for VTP36, VTP37, VTP38, VTP32, RSC12, RAD38, HL02 and
PWR06 are recorded in the plan items themselves.

## 6. Full-cache behaviour (defaults), `qwen27b-tp3-int8-556-long-bf16`, 624,048-token cache

| scenario | result |
|---|---|
| 8 x ~100k distinct prompts (799,760 tokens) + 4 short, at once | all 12 HTTP 200; 6 preemptions (4 forced after `--preempt-wait-ms` 5000, recompute); TTFT 377-564 s; short requests queued ~330 s for a slot; health ok after |
| one 298,667-token prompt (> 262,144) | HTTP 400 `context_length_exceeded`, names the tokens to cut |
| burst of 80 tiny requests | 64 accepted and completed; 16 HTTP 429 `server_overloaded`, "retry after 1s" (queue = 8 x max sessions) |

Defaults: `--max-queued-requests` 8 x `--max-num-seqs`; `--preempt-wait-ms` 5000; preemption by recompute.
