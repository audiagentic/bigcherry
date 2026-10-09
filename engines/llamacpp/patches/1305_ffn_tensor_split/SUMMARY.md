# 1305_ffn_tensor_split

**Status:** evaluated
**Plan item:** QFP09

## What it does

Under `-sm tensor`, `BIGCHERRY_FFN_TS="a,b,c"` gives the FFN/MoE family (routed-expert, shared-expert and dense
FFN up/gate/down weights and biases) its own split vector, independent of `-ts` and of 1303's
`BIGCHERRY_ATTN_TS`. Groups keep their upstream anchors and per-layer rotation. Motivation (QFP09/QFP11): decode
AllReduce arrival skew (~80 us per AR, R9700 last in ~70% of ARs) comes from sizing the R9700's expert share by
VRAM while its bandwidth is ~2/3 of an XTX; this lets expert placement follow bandwidth within VRAM limits.
On Qwen4Exp the shared-expert tensors are MIRRORED upstream, so FFN_TS reaches them only together with 1306
(BIGCHERRY_SHEXP_SPLIT=1); without it 1305 moves the routed and dense FFN only (GPT review req_07434b19f4534cb3).
Attention-family tensors keep 1303's vector (they never take the FFN vector). Activation evidence:
`BIGCHERRY_PATCH_HIT patch=1305_ffn_ts ffn_ts=...`.

## Hardware result (2026-10-04 review)

Parked (QFP09): mechanism works and is cheap; no FFN split vector beat -ts in screens.
