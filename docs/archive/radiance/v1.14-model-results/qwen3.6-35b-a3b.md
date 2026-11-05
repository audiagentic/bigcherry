# Qwen3.6-35B-A3B (FP8)

Latest measured configuration: **Radiance v1.14.0**, one Radeon PRO R9700
(RDNA4) with host-tiered experts. The `qwen3.6-35b-a3b-fp8.rad`
container is 34.6 GiB, with 256 experts (top eight), FP8 block
quantisation, delta-net layers and an MTP head.

## Best recorded throughput

The current-pin test used one stream, 384 generated tokens, three
speculative draft tokens, FP8 KV, a 32K context limit, temperature zero
and three requests per prompt. Results were repeated twice.

| Prompt | Output tokens/s |
| --- | ---: |
| Counting | **218.3–218.5** |
| Code | **212.1–212.2** |
| Story | **170.8–170.9** |

These are prompt-dependent generation rates. The counting result is the
highest of these prompts, **not** a general decode rate. A current-pin
fresh-prefill throughput measurement is not recorded for this setup.

## Configuration and limits

The model is larger than the R9700's 32 GiB VRAM and therefore uses the
host expert pool with `expert_tiered` placement. The recorded run uses
`--tp 1`, FP8 KV, three speculative tokens, a maximum of six sequences
and a 30 GiB configured host expert pool.

Multi-stream MTP with prefix caching is **not qualified** by this
one-stream result. The model's earlier prefix-cache acceptance issue
remains a known risk pending a current-pin multi-stream check; those older
measurements are not presented as current throughput. There is no
matching unmodified Radiance v1.14.0 comparison for this run.

## Evidence

The [RAD37 current-pin check](../../../planning/completed/patching-radiance-rdna3/RAD37.md)
records the `rad37-g35-*` test series and its decode-attention behaviour.
The [dated lab record](../LAB_HISTORY.md)
retains earlier experiments and their limitations. See the
[Radiance overview](../../../reference/engines/radiance/README.md) for the other current models.
