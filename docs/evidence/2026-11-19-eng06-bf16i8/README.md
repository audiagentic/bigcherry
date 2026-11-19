# bf16_i8 KV cache on current main (ENG06, 2026-11-19)

bf16 keys, int8 values with an f16 scale per 32 dims (`rad0038`, `--kv-cache-dtype bf16_i8`). Question (plan
tuning-radiance-engine/ENG06): is it good enough to ship as a named config on the dense TP3 27B and on Flash-Next's
production layout (helpers, live policy)? Build `6cdd2755` (current main, all of the day's serving fixes), 7.0.13-gfxoff
kernel, stock power, GPU lease held for every run. rad0037/38/40/41 were already promoted into `serving-core` (#1033), so
this changes configs, not the patch lifecycle.

## Result

| layout | verdict |
|---|---|
| `qwen27b-tp3-int8-556-long-bf16i8` (new) | ships: decode = bf16, first token fastest of the three, capacity 818,896 tokens (bf16 624,048), needles 18 of 18 |
| `flashnext-4gpu-bf16i8` (new, **not the default**) | serves at **4 sessions**: at the production 8 sessions x 262,144 the engine card is 5.47 GB short at admission |

## Capacity and admission

- TP3 27B, `-556-long`: the plan says 818,896 tokens of context, 3.1 sessions at 262,144 (bf16: 624,048; fp8 about 1.25 M).
- Flash-Next `flashnext-4gpu` at 8 sessions refuses before loading weights: `gpu_room` on the engine card 0000:09:00.0 (R9700,
  32 GB): live 32,171,481,080 bytes against 37,641,672,295 required, short 5,470,191,215 (statics 5.8 GB, experts 2.9 GB,
  stage 1.1 GB). At 4 sessions it fits: 20.67 GiB of KV a card, 1,002,364 tokens, 3.8 sessions at 262,144.

## Flash-Next production layout (helpers, live policy on)

`flashnext-parity.sh measure`, `service.max_sessions=4`, policy on, 451 s warmup, two launches. Launch 2 replayed launch 1's
stored manifest (no planning phase) with the new S8 inventory digest (#1181). Reference: v1.17.0, and the same-day fp8 default
(8 sessions, policy on, `docs/evidence/2026-11-19-flashnext-4gpu-regression/README.md`).

| metric | v1.17.0 | fp8 default (8 sessions) | bf16_i8 #1 | #2 | median | vs fp8 |
|---|---|---|---|---|---|---|
| 256K prefill long7 / long8 tok/s | 2914 / 2901 | 3085 / 3183 | 2975 / 3054 | 2857 / 3048 | 2916 / 3051 | -5.5% / -4.1% |
| 44K document prefill tok/s | 3591 | 3742 | 3605 | 3605 | 3605 | -3.7% |
| cold 27K TTFT s | 7.95 | 8.30 | 8.485 | 8.414 | 8.45 | +1.8% |
| decode 1 stream ms/step | 24.99 | 25.87 | 25.91 | 25.68 | 25.80 | -0.3% |
| 256K-context decode ms/step (long7 / long8) | 35.54 / 27.82 | 28.24 | 28.91 / 29.20 | 28.92 / 28.75 | 28.92 / 28.98 | +2.6% |
| policy commits | 7 | 5, 5 | 5 | 4 | | |

Not comparable (4 session slots instead of 8): the 8-session TTFT and ms/step rows, and the 6-stream decode (36.9 ms/step is
a 4-wide batch). `flashnext-parity.sh compare` against v1.17.0 reports REGRESSION on cold TTFT (+6.3%), 1-stream decode
(+3.2%) and the 8-session rows; the same-day fp8 default has the same shape (+4.3%, +3.5%). The one-stream decode gap of about 6%
between int8 V and E4M3 V seen on two XTX without helpers (VTP44) does not appear here: the helpers dominate the step. The
cost of bf16_i8 against fp8 is prefill at 256K (-4 to -6%: bf16 keys read 36.75 against 24 KiB a page).

## Dense TP3 27B, `-556-long`: bf16_i8 against fp8 and bf16

One fresh start each, ABBA-style (i8, fp8, bf16, bf16, fp8, i8), one stream for TTFT (distinct prompts, no prefix-cache
hits), decode with 512-token prompts and 256 new tokens, 2 reps; engine figures from the server metrics.

TTFT, one stream, seconds (prompt tokens in brackets):

| prompt | bf16_i8 #1 | bf16_i8 #2 | fp8 #1 | fp8 #2 | bf16 #1 | bf16 #2 | i8 vs bf16 | i8 vs fp8 |
|---|---|---|---|---|---|---|---|---|
| 512 [501] | 0.230 | 0.233 | 0.235 | 0.210 | 0.237 | 0.232 | -1.3% | +4.0% |
| 2048 [2038] | 0.755 | 0.768 | 0.785 | 0.702 | 0.785 | 0.746 | -0.5% | +2.4% |
| 8003 [7846] | 2.839 | 2.830 | 2.846 | 2.782 | 2.832 | 2.778 | +1.1% | +0.7% |
| 30000 [29566] | 10.993 | 10.937 | 11.201 | 11.145 | 10.995 | 10.954 | -0.1% | -1.9% |
| 60000 [59615] | 23.522 | 23.435 | 24.325 | 24.328 | 23.644 | 23.625 | -0.7% | -3.5% |
| 100366 [99689] | 42.530 | 42.576 | 45.050 | 45.105 | 43.617 | 43.637 | -2.5% | -5.6% |

Decode (512-token prompts, 256 new tokens, 2 reps; engine figures from the server metrics):

| streams | metric | bf16_i8 #1 | bf16_i8 #2 | fp8 #1 | fp8 #2 | bf16 #1 | bf16 #2 |
|---|---|---|---|---|---|---|---|
| 1 | ms_per_step | 27.63 | 27.61 | 27.98 | 27.97 | 27.62 | 27.62 |
| 1 | tok_per_step | 2.79 | 2.79 | 3.03 | 3.03 | 3.41 | 3.41 |
| 1 | agg_tps | 91.6 | 91.9 | 98.8 | 98.9 | 110.1 | 110.0 |
| 1 | draft acceptance | 0.597 | 0.596 | 0.677 | 0.677 | 0.802 | 0.802 |
| 4 | ms_per_step | 32.29 | 32.31 | 32.52 | 32.63 | 32.18 | 32.35 |
| 4 | tok_per_step | 11.58 | 12.13 | 11.53 | 11.71 | 11.69 | 11.76 |
| 4 | agg_tps | 271.6 | 286.5 | 277.7 | 278.2 | 275.1 | 277.7 |
| 4 | draft acceptance | 0.709 | 0.753 | 0.661 | 0.667 | 0.742 | 0.717 |
| 6 | ms_per_step | 41.00 | 41.82 | 41.71 | 41.57 | 41.39 | 41.43 |
| 6 | tok_per_step | 16.81 | 16.95 | 17.02 | 17.10 | 16.98 | 17.06 |
| 6 | agg_tps | 305.4 | 304.0 | 308.4 | 312.2 | 311.3 | 302.6 |
| 6 | draft acceptance | 0.689 | 0.665 | 0.683 | 0.696 | 0.679 | 0.680 |
| 8 | ms_per_step | 44.28 | 44.30 | 45.44 | 45.57 | 45.57 | 44.59 |
| 8 | tok_per_step | 22.81 | 23.85 | 22.89 | 22.87 | 23.01 | 22.22 |
| 8 | agg_tps | 358.7 | 371.2 | 365.3 | 365.8 | 367.6 | 350.8 |
| 8 | draft acceptance | 0.737 | 0.761 | 0.696 | 0.688 | 0.693 | 0.705 |

bf16_i8 decode costs what bf16 costs (27.6 ms/step) and 1.3% less than fp8; prefill is the fastest of the three at 60K and
above (100K: -2.5% against bf16, -5.6% against fp8). One-stream tokens/step (and so tok/s) differ by arm (2.79 / 3.03 / 3.41):
that is one prompt's draft acceptance, equal across arms at 4, 6 and 8 streams, so it is not a cache effect.

## Needle retrieval

`tools/lab/radiance/needle_probe.py` (new; the VTP44 probe was ad hoc): a passphrase sentence at 10, 50 and 90% depth, 3
repeats, in natural text from `docs/**/*.md`, chat endpoint, thinking off, temperature 0, exact code scored. A different haystack
offset per case; the prompt is rebuilt once with the server's own token count when it lands more than 8% off target.
Prompt tokens actually sent: the 48K class median 49,127 (range 25,312-58,317; the cases are the same in every arm), the 100K
class median 95,020 (92,758-107,343).

| arm | 48K | 100K |
|---|---|---|
| TP3 `-long` + bf16_i8 | 9 / 9 | 9 / 9 |
| TP3 `-long-bf16` | 9 / 9 | 9 / 9 |
| TP3 `-long` (fp8) | 9 / 9 | 9 / 9 |
| `flashnext-4gpu-bf16i8` (shipped config; policy off, no calibration for its name) | 9 / 9 | 8 / 9 |

The Flash-Next miss is a refusal ("I cannot provide that information."), not a wrong code: the same kind of miss the VTP44
probe recorded for bf16 itself. A first needle run used a wrong characters-per-token calibration (prompts of 28-57K tokens)
and is discarded; the probe was fixed (rescale to the server's count) and re-run.

## Shipped configs, end to end

- `qwen27b-tp3-int8-556-long-bf16i8`: a fresh start from the config name reproduces the `--set` arm (`kv_cache_dtype bf16_i8`
  in the manifest): decode 27.59 ms/step, 100K first token 42.51 s, 60K 23.42 s.
- `flashnext-4gpu-bf16i8`: starts from its name with the helpers, serves, needle above. It has its own layout name, so it has no
  live-policy calibration record: it runs with the policy off until calibrated (`bc-serve serve --calibration-run`).

## Not done

- KLD with the helpers on the production layout: not re-run. The KV error does not depend on the helpers (attention is in the
  engine); the numbers stand: long-corpus KLD bf16_i8 0.00269 against bf16 (bar 0.0027), fp8 0.00495 (VTP43/VTP44).
- 256K decode A/B (ENG06 step 3) and the block-key quantisation (step 4); the V-scale broadcast patch (step 1) is a separate
  branch with its own kbench A/B.

## Incident during the runs

A Flash-Next start with the policy on stopped itself 15 s after reaching `running`: the steady-state heat guard saw the heat
proof not match for all six of its fixed 100 ms re-reads. Not provable afterwards; #1183 makes the re-read window follow the
declared histogram period and names the differing field.

`scripts/` has the harness (paths are placeholders): `one.sh` (one launch, sweeps), `tp3.sh`, `fn.sh`, `needle.sh`,
`shipped.sh`, `long.sh` (the step 3 benchmark, not yet run), `analyze.py`. Raw `stream-bench` JSON stayed in the run folders.
