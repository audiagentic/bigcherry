# bc-radiance-1160.0.0

BigCherry on Radiance `v1.16.0` (upstream `769b4c64a671`).

## Changes

- Updated the custom Radiance build to upstream v1.16.0, including plugin ABI compatibility with the existing AMD kernel and remote-MoE components.

## Measurements and validation

- Flash-Next, 15 interleaved rounds against the v1.14.0 BigCherry build: six-stream decode step time was 48.15 ms versus 48.56 ms (-0.83%); one-stream was 24.58 ms versus 24.56 ms (+0.07%). Prefill time to first token was effectively unchanged.
- Text changed between upstream pins. On one single-stream prompt, MTP acceptance fell and engine output was 140.2 rather than 144.5 tokens/s; this is not a same-text throughput comparison.
- The 9/9 patch rebase, build, CPU smoke and two-XTX helper KLD checks passed.

## Limits

- The RX 6900 XT tail-helper layout still fails the KLD p99.9 limit (+0.78 versus +0.5); see RAD46.
- The HyperConn up-mix selector still needs current-pin revalidation. There is no matched stock-engine throughput comparison.
