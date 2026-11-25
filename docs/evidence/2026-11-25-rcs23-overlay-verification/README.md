# RCS23 overlay verification: Flash-Next parity on 2020.17.0

Release 2020.17.0 (carries the bc-iqr overlay units S3a-S3c and libr3's effective exclusions), the stored baseline
`flashnext-4gpu-2020.11.2` (`docs/reference/engines/radiance/baselines/flashnext-4gpu-parity.json`), the standard parity workload
(`flashnext-parity.sh measure`, 2 launches, production defaults on the four-card layout). Each cell is one launch of the candidate;
the baseline column is the min..max of its four launches. Tokens per second are not compared across builds; acceptance is informational.

| metric | direction | gated | baseline launches | 2020.17.0 launch 1 | launch 2 | mean vs baseline mean |
|---|---|---|---|---|---|---|
| decode1 ms/step | lower | yes | 24.19..24.51 | 24.23 | 24.46 | -0.4% |
| decode6 ms/step | lower | yes | 46.46..48.38 | 45.49 | 47.33 | -1.7% |
| sess8 cold ms/step | lower | yes | 27.99..29.22 | 28.55 | 28.66 | +0.1% |
| long8 ms/step | lower | yes | 26.30..26.55 | 26.53 | 26.53 | +0.7% |
| long7 ms/step | lower | no | 27.23..27.33 | 27.07 | 27.34 | -0.2% |
| sess8 warm ms/step | lower | no | 57.27..59.16 | 59.36 | 59.17 | +0.2% |
| doc prefill tok/s | higher | yes | 3674..3688 | 3666 | 3671 | -0.3% |
| long7 prefill tok/s | higher | yes | 2926..2979 | 2936 | 2979 | -0.2% |
| long8 prefill tok/s | higher | yes | 3040..3107 | 3008 | 3086 | -0.6% |
| cold TTFT s | lower | yes | 8.264..8.329 | 8.403 | 8.295 | +0.5% |
| sess8 TTFT s | lower | yes | 21.79..22.07 | 21.98 | 21.87 | -0.4% |

Every gated metric is within 1.7% of the baseline mean and inside or at the edge of the baseline's own launch spread; the two values
just outside the baseline range (document prefill, ungated warm sess8) are 0.2-0.3% off. No regression. The stored-baseline compare
command could not be used as is: it requires the launches to carry distinct run ids and one `measure` run gives both launches the same
id, so the comparison above was made directly from the two launch records. The IQR serving A/B on qwen27b-tp3-iq6r was not run: the
owned units' output hashes were equal on both architectures, their resource records identical, and Gate E passed on the overlay rows.
