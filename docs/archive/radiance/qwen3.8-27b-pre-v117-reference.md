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

## Two RX 7900 XTX (`qwen27b-2xtx`), measured 2026-11-04

Radiance v1.16.0, MXFP4 container, tensor parallel over both cards with the
host-memory `wht6` all-reduce, FP8 KV, 512-token prompts, 256 generated tokens,
temperature 0 (same prompt per row; draft acceptance 0.68 to 0.74 at depth 3).
Aggregate decode tok/s:

| Streams | no speculation | 3 draft tokens | depth 3 step time |
|---|---|---|---|
| 1 | 48.0 | 96.7 | 28.1 ms |
| 2 | 85.8 | 125.9 | 38.6 ms |
| 4 | 126.0 | 168.1 | 52.8 ms |
| 6 | 150.1 | 156.7 | 81.8 ms |

With the gfx1100 MXFP4 row selection of RAD47 (calls of more than 16 rows use the tiled
kernel instead of the skinny rows tuned for gfx1201) the same layout measures 97.5, 125.5,
213.2 and 215.1 tok/s at 1, 2, 4 and 6 streams (step time 27.9, 38.2, 34.4 and 53.3 ms):
unchanged at 1 and 2 streams, +27% at 4 and +37% at 6. Draft acceptance is unchanged and cold
prefill is unchanged (about 2,790 tok/s). Greedy single-stream output is identical to the
previous selection on 8 prompts; with several streams the output differs in the same way two
runs of the previous selection differ from each other (batch composition decides which rows run).

Depth 7 was worse at every stream count measured (91.9, 118.6 and 114.7 tok/s at
1, 2 and 4 streams), so the layout uses 3. The gain shrinks as streams are added
because the step grows into the row band where the selected MXFP4 kernel is slow
on this card (RAD47). Cold prefill of a prompt of about 6,880 tokens: 2.57 s with
`wht6` (2,680 tok/s), 3.18 s with the exact all-reduce, 4.16 s on one card; the
large prefill GEMMs already run at 85 to 107 TFLOP/s per card. One prompt per
row, one run each: the single-stream figures are repeatable (identical across
four prompts earlier), the multi-stream ones are not yet repeated.

## Two RX 7900 XTX, INT8 (W8A8) container, measured 2026-11-04

`qwen3.8-27b-w8a8-df2.rad`: the trunk as int8 rotated by a 128-wide Hadamard (`fwht128`, bf16 scale per row per 128 K, run at
`i8a8h`), an int8 head, and the DFlash2 drafter (MXFP4 linears, 2-bit head copy), converted from the bf16 release with
`data/recipes/q38-27b-w8a8-df2.recipe`. 31.2 GB, of which the drafter is 1.54 GiB (0.77 GiB a rank). Same layout shape as
`qwen27b-2xtx` (two cards, `wht6`, FP8 KV, 8192 x 6, depth 3), with RAD47 and RAD48 in the build.

Before RAD48 the int8 prefill tile ran through the compatibility layer at about 28 TFLOP/s; the native gfx11 form
(`native/r4d_gemm_8a8_prefill.h.rw`) keeps the accumulator in the gfx11 layout through a 128-K block and reads the
activation as a gfx11 row. rad-kbench, N=34816 K=5120: M=2048 25.8 ms to 8.33 ms (3.1x), M=128 1.6 ms to 0.56 ms.
Greedy output is byte-identical to the previous build on four prompts.

| Streams | INT8, no speculation | INT8 + DFlash2 | MXFP4 + DFlash2 (RAD47) |
|---|---|---|---|
| 1 | 35.6 | 83.3 | 97.5 |
| 2 | 65.7 | 132.0 | 125.5 |
| 4 | 112.8 | 205.6 | 213.2 |
| 6 | 149.3 | 201.4 | 215.1 |

Prefill of about 4.5k tokens of text: 1,940 tok/s INT8 against about 2,790 for MXFP4 (it was 950 before RAD48).
INT8 is level with MXFP4 on decode at two or more streams and 15% behind at one stream (twice the weight bytes at the
same bandwidth), and 30% behind on prefill. Its case is quality (the container's mean KL against bf16 is 0.0005 in
the recipe's measurement), not speed. One prompt per row, one run each.

Context, INT8 + DFlash2, fp8 KV, depth 3 (real engine tokens; the bench's token count is words and runs 1.66x under):

| Layout | KV per card | Prompt | Prefill | Decode step | Draft accepted |
|---|---|---|---|---|---|
| 8192 x 6 | 1.68 GiB | 512 | 1,930 tok/s | 31 ms | 3.09 tokens a step |
| 131072 x 2 | 4.32 GiB | 116k | 113 s (1,020 tok/s) | 72 ms | 43 of 63 |
| 262144 x 1 | 4.16 GiB | 248k | 395 s (630 tok/s) | 120 ms | 44 of 60 |
| 262144 x 5 | 8.75 GiB | 248k | 395 s | 121 ms | 44 of 60 |

The engine sizes the pool to the layout's worst case and leaves the rest unclaimed (7.2 GiB a card at 8192 x 6, 96 MiB at
262144 x 5); it reports that the pool could grow to about 21 GiB a card, 1.48M tokens. Speculation holds at depth: the
step grows with attention over the cache, not with drafting.

Drafter on another card (not built): the drafter reads the target's hidden states of the accepted tokens, so a draft
cannot start before its verify ends and there is nothing to overlap with. Drafting plus the four-token verify costs 4.8 ms
over a one-token step (31.0 against 26.2 ms), so moving the drafter off the XTX pair frees at most that and 0.77 GiB a rank,
and adds a host-memory hop of the hidden states each step (no peer-to-peer). A drafter that overlaps with the target
(llama.cpp's separate small model) is a different design from DFlash2.

## Evidence

[RAD37, current-pin model checks](../../planning/completed/patching-radiance-rdna3/RAD37.md)
records `rad37-g27-10310140`: ordered repeated comparisons, no material
change in aggregate throughput after the pin update. The
[dated lab record](LAB_HISTORY.md)
retains kernel investigations and older measurements. See the
[Radiance overview](../../reference/engines/radiance/README.md) for other models.
