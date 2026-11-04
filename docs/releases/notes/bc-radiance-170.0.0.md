# bc-radiance-170.0.0

BigCherry on Radiance `v1.7.0` (upstream `1e968f6e1d86`).

## Changes

- Ported the custom Radiance build and RDNA3 expert helpers to upstream v1.7.0. The remote-MoE hand-off, plugin ABI checks and diagnostic build remain available.

## Measurements and validation

- Flash-Next, R9700 + two RX 7900 XTX: prefill 2,049–2,050 tokens/s at 4K input; decode 122–126 tokens/s (512 output) and 140–141 tokens/s (2K output).
- CPU smoke, plugin ABI checks, strict audit and helper KLD checks passed. The 27B MXFP4 attention comparison also passed.

## Limits

- The helper-row experiment was slower and remains disabled. Performance figures are for the stated model and topology, not a stock-engine comparison.
