# 1306_qwen4exp_shexp_split

**Status:** evaluated
**Plan item:** QFP12

## What it does

With `BIGCHERRY_SHEXP_SPLIT=1`, the Qwen4Exp shared expert (`ffn_{up,gate,down}_shexp`) is tensor-split under
`-sm tensor` using the DSV4 layout (up/gate axis 1, down axis 0, anchored to `ffn_down_shexp.weight`) instead of
falling through to MIRRORED. Upstream mirrors it, so every GPU stores and computes the whole shared expert in every
layer. Split, each GPU does ~1/n of it and its partial output joins the routed experts' partial sum before the same
AllReduce (no extra collective); VRAM for the shared expert also drops to ~1/n per GPU. Requires 1303 (shares its
`<atomic>` include). Found by GPT review req_07434b19f4534cb3. Activation evidence:
`BIGCHERRY_PATCH_HIT patch=1306_shexp_split`.

## Hardware result (2026-10-04 review)

Neutral in screens; parked (QFP09).


## 2026-10-12 b11552 disposition (QFP12 terminal)

**Evaluated, not promoted.** The 2026-10-04 screens measured split 47.2 versus 46.0/46.6 ms/step at ~24K, and 53.6 versus 53.3/53.4 at ~80K. MTP acceptance changed; VRAM saving was ~185 MiB/XTX and ~135 MiB/R9700. No qualified speedup or new GPU measurement exists.

The previous “no extra collective” description is conditional: \`ggml_backend_meta_graph_compute::get_i_delayed\` only merges PARTIAL branches when use-count, exact ADD adjacency, shape/type and compute-flag predicates hold. Graph/reduction telemetry must prove this for the Qwen4Exp scalar-gated shared branch before any revival. Upstream b11552/master has only DSV4 shared-expert placement. Manifest \`patch.toml\` remains evaluated; the Python module's legacy \`STATE="untested"\` is not resolved by this docs-only audit.

No duplicate patch, placement policy or collective. Detailed source, correctness and conditional reopen gates: \`docs/planning/completed/patching-qwen-flash-next/QFP12.md\`.
