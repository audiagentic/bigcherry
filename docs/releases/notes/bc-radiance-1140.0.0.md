# bc-radiance-1140.0.0

BigCherry on Radiance `v1.14.0` (upstream `4a61d7296bc5`).

## Changes

- Updated the serving build to upstream v1.14.0, retaining remote-MoE hand-off, memory-plan admission and the integrated kernel components.

## Measurements and validation

- Flash-Next A/B on R9700 + two XTX helpers + RX 6900 XT tail: one-stream decode changed by +0.14% in step time and six-stream decode by -0.46%; 28 placement-epoch cycles per cycling arm completed without hanging.
- Engine startup canary, memory-plan checks and eight repeated completion comparisons passed. Two-XTX helper offload passed its KLD limits.

## Limits

- The RX 6900 XT tail-helper layout failed the KLD p99.9 limit (+1.58 versus +0.5); see RAD46.
- The HyperConn up-mix selector requires revalidation on this upstream pin. There is no matched stock-engine throughput comparison.
