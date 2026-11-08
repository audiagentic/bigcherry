# Qwen3.8-Flash-Next

The current upstream pin is **Radiance v1.17.0**. The heterogeneous FP8/IQ4R MoE layout uses a Radeon PRO R9700 engine GPU, two RX 7900 XTX helpers and an RX 6900 XT tail.

## Current-pin evidence

A 5 November 2026 cross-pin check used temperature 0, seed 7, 512-token prompts, 512 generated tokens and two alternating passes. On v1.17.0, cold prefill ranged from **3,473 to 3,733 tokens/s**. This is engine-side prefill, not client output. Step timings were recorded for 1, 2, 4 and 6 streams, but aggregate engine-decode and client-output tokens/s were not qualified. Larger context/cache and multi-session throughput remain unqualified.

**Baseline:** v1.16.0 with the same layout and workload. **Best recorded on v1.17.0:** 3,733 prefill tokens/s in the cross-pin check. This is not an unmodified-stock comparison or a demonstrated speedup.

The [historical technical record](../../../../archive/radiance/qwen3.8-flash-next-pre-v117-reference.md) retains the cross-pin conditions and previous-version benchmarks. See also the [v1.14 archive](../../../../archive/radiance/v1.14-model-results/qwen3.8-flash-next.md) and [Radiance overview](../README.md).
