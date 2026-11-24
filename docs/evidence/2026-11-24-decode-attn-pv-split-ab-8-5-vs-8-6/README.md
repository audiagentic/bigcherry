# Decode-attention PV split (2020.8.6) against 2020.8.5: serving regression check

Setup: `qwen27b-tp3`, speculation depth 3, untraced. 2020.8.5 (A) and 2020.8.6 (B) run as three ABBA cycles (A B B A A B B A A B B A), one engine launch per arm under its own lease, seeds 1-3 x 3 repetitions, 6 and 8 streams, 512-token prompts, 192 new tokens, temperature 0. One warm-up pass per launch comes from the harness's first round. Both arms were installed from release archives with the verified install route.

| streams | 8.5, ms/step (sd of 18 runs) | 8.6 | B - A | harness floor (MDD) |
|---|---|---|---|---|
| 6 | 29.32 (1.58) | 29.25 (1.23) | -0.07 ms, -0.2% | 0.85 ms (2.8%) |
| 8 | 31.84 (0.47) | 31.92 (0.56) | +0.08 ms, +0.3% | 0.26 ms (0.8%) |

Arm means (ms/step), 6 streams: A 28.62 31.58 29.52 29.65 28.33 28.23; B 28.86 31.14 29.44 28.48 28.35 29.24. 8 streams: A 32.02 31.75 31.83 31.87 31.77 31.79; B 32.11 32.18 31.35 31.91 32.20 31.77.

Reading: both differences are inside the floor, so there is no regression from the decode-attention change in serving (the standard for this change is regression-only). The change is a kernel-level improvement that the end-to-end harness cannot resolve at these context lengths (about 700 tokens per sequence).
