# VTP51: weight prefetch on against off at 6 and 8 streams (2020.8.5)

The engine issues a read-only weight prefetch beside each decode all-reduce (12 MiB / 8 MiB leading bytes of the next GEMM's weight, delta-net out-projection read limited to four sequences). The tape shows the prefetch op at about 2.6 ms
a pass, but that is the op's own GPU time, not its cost on the critical path. This A/B measures what the mechanism is worth in serving.

Setup: `qwen27b-tp3` on 2020.8.5, speculation depth 3, untraced; three ABBA cycles (P F F P P F F P P F F P; P = `weight-prefetch: on` as shipped, F = `weight-prefetch: off`, the engine log of the F arms
prints `--weight-prefetch off` and declares 1712 ops instead of 1887), one engine launch per arm under its own lease, seeds 1-3 x 3 repetitions, 512-token prompts, 192 new tokens, temperature 0.

| streams | on, ms/step (sd of arm means, 6 arms) | off | off - on (s.e.) | tokens/step on / off |
|---|---|---|---|---|
| 6 | 30.41 (0.62) | 30.49 (0.37) | +0.08 ms, +0.3% (0.29) | 16.33 / 16.00 |
| 8 | 31.89 (0.22) | 32.20 (0.14) | +0.31 ms, +1.0% (0.11) | 18.33 / 18.18 |

Arm means (ms/step), 8 streams: on 31.84 31.54 31.90 32.14 31.82 32.11; off 32.15 32.34 31.96 32.20 32.19 32.35. 6 streams: on 29.73 29.71 30.81 30.16 31.05 30.99; off 30.09 30.28 30.53 31.04 30.80 30.22.

Reading: the whole prefetch mechanism is worth +0.31 ms (1.0%, about 3 standard errors) at 8 streams and nothing measurable at 6 streams (the 6-stream floor is about 0.85 ms). It is far under the 2.6 ms the tape suggested and
under the 1 ms threshold set for opening a derived-cap item for the delta-net out-projection read: extending that read beyond four sequences can at best recover part of 0.3 ms. VTP51 closes as NO-GO.
