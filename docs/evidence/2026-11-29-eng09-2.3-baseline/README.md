# ENG09: performance baseline of the 2.3 build

Plan item ENG09. Release bc-radiance-2030.5.0 (upstream Radiance v2.3.0), as installed. `bin/radiance` sha256
`c4e21814…73ed3`, `bc-serve` sha256 `a9ea4c31…b047a` (full values in `summary.json`). Measured 2026-11-28 23:32 to
2026-11-29 00:19 AEDT, one gpu lease per run, released between runs.

Power: every run's plan is profile `stock`, mode `observe`; no power setting was written. The efficient profile is not
applied anywhere yet (it is a pending owner decision).

The numbers are in `summary.json`. Raw run folders were not kept; the Flash-Next launch records are embedded in the
baseline file.

## Flash-Next, four cards (flashnext-4gpu)

`tools/lab/radiance/flashnext-parity.sh measure`, two launches, stock power. The new baseline is
`docs/reference/engines/radiance/baselines/flashnext-4gpu-parity-2.3.json`. Two tool fixes were needed first (#1773):

- the 8-session covariate counts mixed steps;
- each launch gets its own run id when the stored plan is reused.

Because the covariate definition changed, `compare` refuses the old baseline.

| Metric | 2020.11.2 baseline | 2.3 (2030.5.0) |
|---|---|---|
| decode, 1 stream, ms/step | 24.43 | 24.48 |
| decode, 6 streams, ms/step | 47.22 | 47.44 |
| decode, 8 sessions (cold), ms/step | 28.58 | 28.64 |
| decode, long context (seed 8), ms/step | 26.35 | 26.42 |
| cold prefill TTFT (27K tokens), s | 8.30 | 8.33 |
| 8-session TTFT (13.6K each), s | 22.0 | 21.9 |
| document prefill (44K), tokens/s | 3,678 | 3,675 |
| long-context prefill (256K), tokens/s, seeds 7 / 8 | 2,963 / 3,067 | 2,974 / 3,113 |
| draft acceptance, 1 / 6 streams | 0.60 / 0.67 | 0.63 / 0.64 |

Every gated metric is within 1.5% of the previous baseline, so 2.3 brings neither a regression nor a gain on this
layout.

The cold 8-session round now has about 46 mixed steps: a session's prefill runs beside the others' decode. Its new
covariate (4.6 decode tokens per decoding step) is therefore not comparable with the old 26.9 pure-decode figure.

## Qwen3.8-27B, TP3 on XTX + XTX + R9700

The models were started from a copy of the release config, with its own port and the power profile set to `stock`. The
work root was the bc-model store. Measurements:

- TTFT: one stream, three seeds per length, one round each (no prefix-cache repeat).
- Decode: `bigcherry stream-bench`, 512-token prompts and 256 tokens, seeds 7, 21 and 42.
- Long session: one prompt of 255,743 tokens.

Values are medians over seeds.

### Capacity

| Config | Model | Pool tokens a rank (own split) | Common split |
|---|---|---|---|
| qwen27b-tp3-long | IQ8R | 1,442,463 | 752,159 |
| qwen27b-tp3-long-iq6r | IQR-V4 6-bit (ours) | 1,520,927 | 863,775 |

Both serve the full 262,144-token session.

### Prefill (single stream)

| Prompt tokens | IQ8R TTFT | IQ8R tokens/s | IQ6R TTFT | IQ6R tokens/s |
|---|---|---|---|---|
| 516 | 246 ms | 2,094 | 281 ms | 1,836 |
| 2,024 | 791 ms | 2,558 | 762 ms | 2,655 |
| 8,129 | 3.18 s | 2,558 | 3.11 s | 2,617 |
| 32,460 | 13.7 s | 2,364 | 13.4 s | 2,430 |
| 255,743 (long session) | 174.8 s | 1,463 | 172.2 s | 1,486 |

### Decode

| Streams | IQ8R tokens/s | IQ8R ms/step | IQ6R tokens/s | IQ6R ms/step |
|---|---|---|---|---|
| 1 | 123.1 | 20.4 | 107.3 | 23.4 |
| 2 | 215.7 | 21.8 | 178.4 | 24.6 |
| 4 | 317.3 | 25.7 | 282.3 | 27.4 |
| 8 | 408.3 | 36.5 | 376.7 | 37.8 |
| 1, at 255,743 tokens | 2.80 tokens/step | 33.9 | 2.52 tokens/step | 36.9 |

At 8 streams the worst TTFT is 1.77 s for IQ8R and 1.75 s for IQ6R. The p95 event gap is 40 and 42 ms.

Draft acceptance varies by prompt:

| Streams | IQ8R | IQ6R |
|---|---|---|
| 1 | 0.59–0.65 | 0.58–0.73 |
| 8 | 0.67–0.72 | 0.58–0.64 |

## Reading

- Flash-Next on 2.3 matches the 2020.11.2 baseline.
- Our 6-bit IQR-V4 decodes slower than IQ8R on 2.3 despite fewer weight bytes:
  - +15% ms/step at one stream;
  - +3.5% at eight streams;
  - +9% at 256K.
  It also gains only 5% capacity, and its prefill is level or slightly faster from 2K tokens up. This is the expected
  starting point for the IQ6R optimisation; the gap is in the decode step, not in prefill.
- Not covered yet: the upstream 4-bit (q4) and upstream 6-bit (q6) models. Their vocabulary heads are cut for one or two
  ranks, which this TP3 start refuses. They follow in a later PR.
