# qwen3.6 35b a3b

The 34.6 GiB FP8 MoE container routes eight of 256 experts and uses host-tiered placement on the 32 GiB R9700. Historical tests used FP8 KV, three speculative draft tokens and a 32K context limit. Multi-stream MTP and prefix-cache acceptance remain unqualified on the current pin.

## Current pin

The current upstream pin is Radiance v1.17.0. Fresh qualified prefill throughput (tokens/s), aggregate client output throughput (tokens/s), aggregate engine decode throughput (tokens/s), and a like-for-like unmodified-stock comparison are unavailable for this model on the current pin. No previous-version result is presented as a current measurement.

## Configuration and limits

See the [archived v1.14.0 measurements](../../../../archive/radiance/v1.14-model-results/qwen3.6-35b-a3b.md) for model quantisation, hardware layout, prompt, stream count, context, dates, metric type and quality limitations. Those historical results are not evidence of v1.17.0 performance. The [Radiance overview](../README.md) describes the current implementation.
