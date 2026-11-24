# Serving A/B: release 2020.11.1 (IQR wide tile, 17-64 rows) against 2020.11.0

Result: level everywhere. No arm is slower than its previous-release arm beyond the harness floor (0.85 ms at six streams, 0.26 ms at eight).

Method: both releases installed side by side, order old/new/new/old (ABBA) per container, one launch per arm, three warm-up requests, seeds 1-3 with three repetitions per stream count, 512-token prompts, 192 new tokens, temperature 0. Time to first token is the median of five requests (three at 8192 tokens). Single-stream temp-0 text is byte-identical in all twelve launches.

## IQR-V4 on three ranks (qwen27b-tp3-iq6r), ms/step and tokens/step

| streams | 2020.11.0 (two launches) | 2020.11.1 (two launches) |
|---|---|---|
| 1 | 21.43 / 21.53 (2.93) | 21.54 / 21.57 (2.93) |
| 6 | 29.80 / 29.89 (13.6-13.7) | 29.98 / 29.84 (13.6-13.8) |
| 8 | 32.95 / 32.94 (17.2-17.3) | 32.83 / 32.68 (16.8-17.1) |

First token at 128 tokens: 121.0 / 118.9 ms before, 118.5 / 118.8 ms after. At 2048 tokens: 780.6 / 776.5 before, 722.0 / 774.8 after (spread across launches about 8%, no consistent move).

## IQR-V4 on the long layout (qwen27b-tp3-iq6r-long, 262144-token sessions)

| streams | 2020.11.0 | 2020.11.1 |
|---|---|---|
| 1 | 21.51 / 21.52 | 21.53 / 21.52 |
| 6 | 32.23 / 32.17 | 32.14 / 32.37 |
| 8 | 36.97 / 36.81 | 36.83 / 36.82 |

First token at 128 / 2048 / 8192 tokens: 123.9 / 765.2 / 1627 ms and 126.1 / 733.6 / 1634 ms before; 118.7 / 728.9 / 1639 ms and 119.2 / 728.0 / 1635 ms after. The 8192-token prompt is level to 0.5%: the prefill chunk the engine issues is larger than the wide tile's 17-64 rows, so the wide tile does not sit on this row. The small 2048-token gain on the long layout (about 4%) is within the launch spread.

## IQ8R on three ranks (plain path, qwen27b-tp3)

| streams | 2020.11.0 | 2020.11.1 |
|---|---|---|
| 1 | 20.23 / 20.22 | 20.24 / 20.23 |
| 6 | 28.54 / 28.12 | 28.30 / 29.57 |
| 8 | 31.20 / 31.64 | 31.28 / 31.30 |

Six streams moves with the number of tokens accepted per step in that launch (14.3-16.0), the usual spread of this row; first token at 128 / 2048 tokens is 106.3 / 107.0 and 792.7 / 781.3 ms before, 107.6 / 103.1 and 776.0 / 792.3 after. The plain path is level, as expected for a change confined to the IQR tile.

## Reading

The wide tile's gain is at 17-64 rows of an IQR layer (replay, byte identity: its own PR). In whole-model serving at 1, 6 and 8 streams and at prompts to 8192 tokens, the rows reaching it are not on the critical path, so the serving rows neither gain nor lose: no regression, no measurable gain.
