# Qwen3.8-Flash-Next

The current upstream pin is **Radiance v2.2.0**. The heterogeneous
FP8/IQ4R MoE layout uses an R9700 engine GPU, two RX 7900 XTX helpers
and an RX 6900 XT tail.

## Current-pin measured prefill

The [RUM09 paired release gate](../../../../evidence/radiance-latest-bump/RUM09-GATE-D.md)
used the flashnext-4gpu profile, 262,144-token context, eight sessions,
stock power profile and matched prompts, seed and topology. It recorded
two launches per arm in November 2026.

| Workload | Baseline: v1.17.0 prefill tokens/s | Best recorded: v2.2.0 prefill tokens/s |
| --- | ---: | ---: |
| Document prefill | 3,254 | **3,662** |
| Long-7 prefill | 2,207 | **2,453** |
| Long-8 prefill | 2,099 | **2,430** |

These are **engine-side prefill** figures, not client-output throughput.
The gate does not provide qualified aggregate engine-decode or
client-visible output tokens/s, nor a like-for-like unmodified-stock
comparison. The reported decode step latency cannot be presented as
throughput. Do not extrapolate these results to different cache,
stream-count or model configurations.

Older cross-pin results are retained in the
[historical technical record](../../../../archive/radiance/qwen3.8-flash-next-pre-v117-reference.md)
and [v1.14 archive](../../../../archive/radiance/v1.14-model-results/qwen3.8-flash-next.md).
