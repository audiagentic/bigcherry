# bc-radiance-180.0.0

BigCherry on Radiance `v1.8.0` (upstream `7a7707a95559`).

## Changes

- Ported the custom build and AMD kernel components to upstream v1.8.0. Added the FFmpeg 6 MP4-writer compatibility patch.

## Measurements and validation

- Hardware acceptance passed on the two-XTX Flash-Next layout: prefill 3,389 tokens/s (4K input), 3,439 tokens/s (1K input); decode 146 tokens/s (512 output).
- Helper KLD was within its recorded acceptance limits. Source, plugin ABI and CPU smoke checks passed.

## Limits

- These are workload-specific acceptance results compared with stored baselines, not a like-for-like stock Radiance speedup.
