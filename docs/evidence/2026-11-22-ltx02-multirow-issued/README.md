# LTX02: checkpoint-backed 16/64-row attention

**Host and mixed AQL issue pass; TP1 parity remains open.** The existing C06 RadCtx/Builder self-attention graph now supports genuine 16 and 64 token rows: quantization, QKV, all-head Q/K normalization, RoPE, attention, exact reductions and output projection. Video/audio positions and cu-seqlens derive from the row count. The original single-row mode is preserved. No new loader, collective, graph, or production route.

## Verification

- Source-pinned native host build, low-priority single worker: **13/13 CTests PASS**, including both new checkpoint-issued multirow cases.
- Leased mixed AQL (two gfx1100 and one gfx1201), both legal 8/8/16 and 12/12/8 head splits and video/audio block zero: **16 and 64 rows PASS** in both physical GPU orders 0/1/2 and 2/0/1. All ranks within every graph produce identical finite, nonzero BF16 output.
- Repeated runs on an identical GPU order have byte-identical outputs. All four 16-row graph outputs are also bitwise identical across GPU orders.

## Unresolved 64-row GPU-order precision

The first divergent captured stage is `attention-after-gate`. Per-rank input quant codes/scales, QKV projections, head gates and Q/K-normalization/RoPE tensors are byte-identical across GPU orders. Differences are reproducible by GPU order rather than run-to-run noise. They may reflect architecture-dependent attention rounding, but acceptance is NOT established.

| 64-row case | BF16 elements differing | Relative L2 | Maximum absolute |
| --- | ---: | ---: | ---: |
| Audio 8/8/16 | 351 / 131,072 | 0.01748% | 0.00390625 |
| Audio 12/12/8 | 360 / 131,072 | 0.01946% | 0.015625 |
| Video 8/8/16 | 796 / 262,144 | 0.01466% | 0.03125 |
| Video 12/12/8 | 796 / 262,144 | 0.01541% | 0.0625 |

These measured differences are **not an approved model-quality threshold**. Next: compare attention intermediates against captured TP1 activations and the pinned per-device numerical reference. If unacceptable, correct the owning generic attention kernel, not an LTX-specific workaround. RUM14 retains resident-weight loading; LTX03 retains non-self attention and full 48-block issue. No TP3 serving, release, full-video quality or fault/cancel acceptance is implied. Raw AQL receipts and captures remain outside Git.
