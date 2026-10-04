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
