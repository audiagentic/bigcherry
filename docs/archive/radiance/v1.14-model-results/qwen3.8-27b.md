# Qwen3.8-27B (MXFP4)

Latest measured configuration: **Radiance v1.14.0**, one Radeon PRO R9700
(RDNA4), `qwen3.8-27b-mxfp4.rad` (19 GB), FP8 KV cache, 32K maximum
context, prefix cache enabled and three speculative draft tokens.

## Best recorded throughput

The current-pin multi-stream test used 512-token prompts and 128 generated
tokens per request. These are **aggregate client output tokens/s**;
each range covers two repeated comparisons at the same version.

| Concurrent streams | Output tokens/s |
| ---: | ---: |
| 1 | **79.5–79.7** |
| 2 | **137.9–138.6** |
| 4 | **205.7–206.0** |
| 6 | **245.1–248.2** |

Six concurrent requests deliver about three times the aggregate throughput
of one request, at a lower rate per session. The engine's internal decode
counter is higher than client output: around 490 tokens/s at six streams.
Do not substitute that counter for generated text throughput.

**Prefill:** a fresh, controlled prefill result on v1.14.0 is **not
recorded** for this configuration. Previous-version prefill measurements
and two-XTX tensor-parallel tuning are excluded from the current result.

## Configuration and limits

This is a dense model, so it runs on the engine card without remote expert
helpers. The alternative two-RX-7900-XTX path uses `libr3`, tensor
parallelism and `wht6` cross-card communication, but there is no matched
v1.14.0 performance result to rank it against the R9700 setup here.

To exercise the alternative layout without hardcoded card indices, use
`qwen27b-2xtx` in `engines/radiance/layouts.toml` with
`bash tools/lab/radiance/launch-flashnext-best.sh start --layout qwen27b-2xtx`
after sourcing the host environment and acquiring the GPU lease. This selects
the two inventoried `gfx1100` cards, TP=2, `libr3`, `wht6` and host-memory
cross-card transport (`--p2p off`); there are no remote-MoE helpers. The
interim calibrated `bc-serve` plan-check does not cover dense TP layouts; the
launcher instead requires a fresh successful memory-plan measurement. Hardware
startup, memory headroom, throughput and output quality remain unverified on
the current build.

The current multi-stream run uses the model's MXFP4 container rather than
the separate FP8 + DFlash2 container. The 27B DFlash2 drafter at depth 7
has not been remeasured in this configuration on the current pin. There is
no like-for-like stock Radiance comparison or full-precision reference
quality measurement for this run.

## Evidence

[RAD37, current-pin model checks](../../../planning/completed/patching-radiance-rdna3/RAD37.md)
records `rad37-g27-10310140`: ordered repeated comparisons, no material
change in aggregate throughput after the pin update. The
[dated lab record](../LAB_HISTORY.md)
retains kernel investigations and older measurements. See the
[Radiance overview](../../../reference/engines/radiance/README.md) for other models.
