# Regression sweep: release 2020.9.2 against 2020.8.7 (qwen27b and flashnext)

Result: no regression in any arm. Method: both releases installed side by side, arms interleaved old/new/old/new, one launch per arm, same prompts and seeds, three warm-up requests, three seeds with repetitions per stream count. Host shared with other sessions (load average 3-11 during the runs).

## qwen27b-tp3, IQ8R container (ms/step, mean of seeds; two launches per release)

| streams | 2020.8.7 | 2020.9.2 |
|---|---|---|
| 1 | 20.21 / 20.26 | 20.25 / 20.25 |
| 6 | 29.64 / 29.04 | 27.97 / 30.07 |
| 8 | 31.80 / 32.34 | 31.25 / 31.33 |

Six streams differs by up to 1.1 ms between launches of the same release and the order flips between repeats; that is inside the 0.85 ms harness floor plus launch spread. Eight streams is level to 0.5 ms faster. The 128-token and 2048-token time to first token agree to 1% in all four launches (about 109 ms at 128 tokens).

## qwen27b-2xtx, IQ8R container (1, 4, 6 streams)

| streams | 2020.8.7 | 2020.9.2 |
|---|---|---|
| 1 | 24.40 / 24.40 | 24.39 / 24.40 |
| 4 | 27.66 / 27.68 | 27.33 / 27.40 |
| 6 | 31.27 / 31.50 | 30.06 / 30.96 |

## qwen27b-tp3, IQR container with the TP3 head (shipped head against shipped head plus this series)

An earlier first attempt used a comparison root that also carried an experiment that was later rejected; it is not reported. The corrected pair (2020.8.7 with the owned weighted head only, against the same release with the head tile, accumulate-once and native-fragment changes):

| streams | before | after |
|---|---|---|
| 1 | 29.97 / 30.16 | 21.58 / 21.56 |
| 6 | 55.97 / 61.29 | 29.93 / 30.08 |
| 8 | 57.81 / 59.98 | 32.96 / 33.04 |

Text at one stream and at eight is identical in all four launches.

## flashnext-4gpu parity harness (one launch per arm, ABAB)

| row | 2020.8.7 | 2020.9.2 |
|---|---|---|
| 1 stream ms/step | 24.4, 24.1 / 24.3, 24.1 | 24.6, 24.4 / 24.3, 24.0 |
| 6 streams ms/step | 47.3, 45.9 / 46.3, 45.8 | 47.8, 47.4 / 47.3, 47.8 |
| 8 sessions, decode ms/step | 28.4 / 28.5 | 28.6 / 27.7 |
| long-context 7 and 8 | 26.5-28.0 / 25.8-27.2 | 26.6-28.0 / 25.8-27.2 |
| cold fill time to first token | 8.29 / 8.38 s | 8.29 / 8.38 s |

The stored parity baselines are in a record format the compare command of this harness no longer accepts ("unsupported baseline format"); the comparison above is therefore release against release in the same session. The six-stream row of the new release reads 0.5-1 ms above the old one in one launch and level in the other, inside the launch spread.

## Start time of flashnext-4gpu

Time from launch to running, four launches interleaved, heat files present: 85.3 / 83.2 / 85.3 / 85.2 s (old, new, old, new), host load average 3-11. The 287 s seen once earlier was therefore load (a concurrent build and measurement), not the release.

## Text identity

Single-stream temp-0 text is byte-identical in all arms of every row. Concurrent 8-stream texts differ between launches of the same release (batch composition depends on arrival timing), so they carry no release signal.
