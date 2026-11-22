# LTX03 — native full-depth self-attention declaration gate

**Scope:** one additive CTest over the existing model-backed C06 Builder and patched LTX2 Attn::declare; no new graph, loader, collective, or kernel path.

The test derives the DiT block count from `transformer.num_layers` in the real IQ8R checkpoint, uses the existing video/audio self-attention declaration for each block, and checks both legal 8/8/16 and 12/12/8 head splits across three ranks. Per-rank validation remains in `run_self_attn`: checkpoint source names and geometry, native IQ8R group and QKV banks, all-head RMS declaration, exact collective ownership, post-reduce bias, RoPE head offsets, and kernel-selected host rows.

## Host gate

- `ltx03_all_block_self_attention_declarations`: PASS, 576 native declarations (48 blocks x 2 modalities x 2 splits x 3 ranks).
- Existing model-backed host CTest suite: **11/11 PASS**, including the prior C05/C06 checkpoint issue and C06 weight-share tests.
- Build: native host reference fixture, single compile worker at low priority.

## Remaining LTX03 work

This test checks **declaration only**, not GPU execution, and only two of six attention variants. It neither stages every block's weights nor proves real TP1 input parity. LTX03 still needs text/cross-modal attention, FFN, modulation, residuals, model-specific ordered issue, all-block actual execution, and numerical comparisons. RUM14 remains the sole owner of generic resident-weight loading; do not add a second LTX loader. LTX02 remains responsible for captured TP1-activation parity. No TP3 serving enablement or release admission is implied.
