# Dense Qwen3.8-27B TP3 no-regression check on current main (2026-11-19)

Question (user, 2026-11-19): do today's serving changes (#1162 calibration policy refusals, #1169 policy key, #1172
per-source-path component trees, #1174 heat guard, #1175 compile retry, #1176 receipt, #1177 replay projection, #1178
parity harness) regress the dense TP3 configs? Dense TP3 has no live policy; the changes touch `bc-serve` and build
tooling, not the engine.

Host: the 7.0.13-gfxoff kernel, 2 x 7900 XTX + R9700 (split 5,5,6), stock power, one fresh `bc-serve serve CONFIG`
per launch from the same build (`6cdd2755`, current main), GPU lease held throughout.

## Method

- **Workload** (`scripts/one.sh`): one warm-up request (not measured), then one single-stream request for each of 512, 2048,
  8003, 30000, 60000 and 100366 prompt tokens with a different seed each (no prefix-cache hits), 8 new tokens, TTFT
  read from the stream; then decode at 1, 4, 6 and 8 streams with 512-token prompts, 256 new tokens a stream, 2 reps.
  `stream-bench` calibrates the prompt to the server's tokenizer; it lands 0.7-2% under the target (the token counts are
  in the tables), so compare per token, not per target.
- **Reported separately:** TTFT/prefill (seconds, prefill rate) and decode (engine ms/step, tokens/step, chunks/s, tok/s).
  `chunks_per_s` = stream events over the decode window (wall minus the slowest first token); it is the quantity the VTP37
  and RAD52 evidence call "chunks/s".
- **Control arm.** The recorded references are from other days and harnesses, so alone they cannot separate `bc-serve`
  from drift. Today's changes leave the engine bits unchanged, so the clean test is the same engine with the `bc-serve`
  from before today (`4fabff91f`, the S16 commit, built in scratch and swapped into a copy of the build) against the
  current one, in one session and sandwiched: new bf16, new fp8, old bf16, old fp8, new fp8, new bf16
  (`scripts/all.sh`, `scripts/all2.sh`). Run-to-run: the two new launches of each config give the spread.
- Every launch used its own run directory (no stored-manifest replay), because replay failed (see below).

## Results

### qwen27b-tp3-int8-556-long-bf16

TTFT, one stream, seconds (prompt tokens in brackets):

| prompt | l1 new | l5 old | l8 new |
|---|---|---|---|
| 512 [501] | 0.209 | 0.227 | 0.236 |
| 2048 [2038] | 0.702 | 0.774 | 0.782 |
| 8003 [7846] | 2.794 | 2.818 | 2.843 |
| 30000 [29566] | 10.998 | 11.015 | 11.020 |
| 60000 [59615] | 23.641 | 23.622 | 23.637 |
| 100366 [99689] | 43.634 | 43.674 | 43.556 |

Decode (512-token prompts, 256 new tokens, 2 reps): engine ms/step, tokens/step, chunks/s per start, and tok/s aggregate:

| streams | metric | l1 new | l5 old | l8 new |
|---|---|---|---|---|
| 1 | ms_per_step | 27.61 | 27.64 | 27.62 |
| 1 | tok_per_step | 3.41 | 3.41 | 3.41 |
| 1 | chunks_per_s | 36.6 | 36.6 | 36.6 |
| 1 | agg_tps | 110.2 | 110.3 | 110.0 |
| 4 | ms_per_step | 32.18 | 32.21 | 32.12 |
| 4 | tok_per_step | 11.71 | 11.71 | 11.55 |
| 4 | chunks_per_s | 113.9 | 113.6 | 112.9 |
| 4 | agg_tps | 278.6 | 281.2 | 275.1 |
| 6 | ms_per_step | 41.00 | 41.09 | 40.81 |
| 6 | tok_per_step | 16.73 | 16.74 | 16.51 |
| 6 | chunks_per_s | 136.3 | 135.9 | 135.4 |
| 6 | agg_tps | 311.9 | 311.2 | 300.0 |
| 8 | ms_per_step | 44.46 | 45.96 | 44.66 |
| 8 | tok_per_step | 22.25 | 23.37 | 22.16 |
| 8 | chunks_per_s | 164.9 | 168.6 | 163.8 |
| 8 | agg_tps | 363.7 | 357.1 | 350.8 |

### qwen27b-tp3-int8-556-long

TTFT, one stream, seconds (prompt tokens in brackets):

| prompt | l2 new | l6 old | l7 new |
|---|---|---|---|
| 512 [501] | 0.233 | 0.231 | 0.231 |
| 2048 [2038] | 0.755 | 0.779 | 0.785 |
| 8003 [7846] | 2.797 | 2.843 | 2.846 |
| 30000 [29566] | 11.153 | 11.130 | 11.181 |
| 60000 [59615] | 24.272 | 24.289 | 24.284 |
| 100366 [99689] | 45.038 | 45.105 | 45.099 |

Decode (512-token prompts, 256 new tokens, 2 reps): engine ms/step, tokens/step, chunks/s per start, and tok/s aggregate:

| streams | metric | l2 new | l6 old | l7 new |
|---|---|---|---|---|
| 1 | ms_per_step | 27.96 | 27.95 | 27.95 |
| 1 | tok_per_step | 3.02 | 3.02 | 3.02 |
| 1 | chunks_per_s | 36.1 | 36.1 | 36.1 |
| 1 | agg_tps | 98.4 | 98.3 | 98.4 |
| 4 | ms_per_step | 32.69 | 32.70 | 32.70 |
| 4 | tok_per_step | 11.77 | 11.75 | 11.75 |
| 4 | chunks_per_s | 120.7 | 120.7 | 120.6 |
| 4 | agg_tps | 276.6 | 272.9 | 272.4 |
| 6 | ms_per_step | 41.43 | 41.83 | 42.17 |
| 6 | tok_per_step | 16.68 | 16.86 | 17.07 |
| 6 | chunks_per_s | 135.8 | 137.2 | 136.0 |
| 6 | agg_tps | 301.6 | 298.1 | 300.5 |
| 8 | ms_per_step | 44.73 | 45.10 | 44.80 |
| 8 | tok_per_step | 22.84 | 22.70 | 22.60 |
| 8 | chunks_per_s | 165.2 | 165.3 | 164.8 |
| 8 | agg_tps | 362.5 | 354.1 | 356.0 |

## Against the recorded references

| quantity | recorded | today (new bc-serve) | |
|---|---|---|---|
| bf16 prefill rate, 100,366-token class | RAD52: 100,366 tok in 43.91 s = 2286 tok/s | 99,689 tok in 43.63 / 43.56 s = 2285 / 2289 tok/s | equal |
| bf16 TTFT 60k / 30k / 8k (s) | 24.08 / 11.18 / 2.891 | 23.64 / 11.00 / 2.79-2.84 (59,615 / 29,566 / 7,846 tokens) | equal per token |
| fp8 1-stream decode, chunks/s | VTP37 main: 36.26 | 36.13 / 36.14 | -0.4% |
| bf16 1-stream decode, chunks/s | RAD52: 33.72 | 36.6 / 36.6 | +8.6% (later engine work; not a regression) |
| fp8 4 / 6 / 8 streams, chunks/s | VTP37: 101.3 / 107.5 / 134.5 | 120.7 / 135.9 / 165 | higher, **not like-for-like**: the recorded prompt length and window are not stated |

## Findings

1. **No regression.** The same engine with the `bc-serve` from before today and the current one agree within run-to-run
   spread on every decode and prefill number; decode ms/step differs by <1% (1, 4, 6 streams) and TTFT at 8k and above
   by <1%. Sub-second TTFT (512, 2048 tokens) varies by up to 10% between launches of the same arm (0.209 vs 0.236 s for
   the two new bf16 launches), so no difference there is attributable to `bc-serve`.
2. **Defect found, not a performance regression:** a second start of an unchanged config in the same run directory,
   20 s after the first stopped, is refused: `bc-serve replay S8: BC_CFG_CALIBRATION_INVALID ... p1.scope: current measured
   model/build/owner-map/service/inventory scope differs from S8 evidence` (the two failed launches of the first run;
   both configs). Cause: `v2/p1_producer.cpp` and `s8_replay.cpp` both set `inventory_digest = sha256(facts.gpus.dump())`,
   and the detection holds live `vram_used_mib` / `vram_available_mib` (and the negotiated link state), which drift by tens
   of MiB within 20 s as the previous engine's pages are released. Replays worked only when VRAM happened to be
   bit-equal (the earlier Flash-Next 2-launch runs).
   Fix: one `gpu_inventory_digest()` over identity fields only (architecture, BDF, HIP ordinal, total VRAM, gfx target, KFD
   node, NUMA node, SIMD count, DRM minor, max link). Test first: a replay with drifted VRAM and link state must verify, a
   changed architecture or total VRAM must still refuse (`serving_test.cpp`). Serving CTest 34/34.
3. **Hardware proof of the fix** (the branch's `bc-serve` swapped into a copy of the build): dense
   `qwen27b-tp3-int8-556-long-bf16`, two consecutive starts in one run directory, 20 s apart: the first start measured its
   plan and reached `running` after 518 s; the second replayed the stored manifest and reached `running` after 51 s
   with no planning or measuring phase (`scripts/all3.sh`). A `flashnext-4gpu` pair through the same replay path
   (`scripts/fn.sh`) was started after this page was written: a fresh run directory makes the first start hash the whole
   114 GiB model (the identity is cached per run directory), so it is not part of this evidence; its result is a follow-up.

Run folders and raw `stream-bench` JSON were in the run folder under `<work-root>` and are not copied (size);
the scripts above regenerate them. Paths in the scripts are placeholders.
