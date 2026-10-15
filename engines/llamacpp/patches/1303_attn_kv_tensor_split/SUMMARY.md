# 1303_attn_kv_tensor_split

**Status:** validated
**Plan item:** QFN03

## What it does

Under `-sm tensor`, `BIGCHERRY_ATTN_TS="a,b,c"` gives the full-attention family (q/k/v/qkv weights and biases,
q/k norms, sinks, gate, attn_output, KV cache) its own split vector, independent of `-ts`, so KV/attention
placement is decoupled from expert-weight placement and every GPU can be filled at long context.
`BIGCHERRY_ATTN_ROTATE=0` disables the per-layer rotation for that family (pin whole KV heads to chosen devices).
Recurrent (GatedDeltaNet) layers' attn_qkv/attn_gate and their state stay on `-ts` (anchored to ssm_out);
indexer tensors and caches stay mirrored as upstream. All attention-family members use the same vector and
rotation, preserving GQA grouping and the meta backend's split-state equality. Design: GPT req_0f390645741c4413.
Activation evidence: `BIGCHERRY_PATCH_HIT attn_ts=... attn_rotate=...`. Qwen4Exp only: any other architecture fails to
load with a clear error (Gemma 4 produced garbage silently with the split on, 2026-10-04 smoke).

## Hardware result (2026-10-04 review)

Production profile v2 (BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0): f16/f16 KV at 240K; profile ABBA +3% at ~10K, +7% at ~80K; greedy identical. Fails closed on non-qwen4exp (Gemma 4 smoke).

## QFP07 / BCOP207: physical two-head split admission (2026-10-14)

For the observed two-KV-head Flash-Next GGUF on three Meta ranks, 1303's pinned cumulative split rounds to whole KV heads. A nonzero three-way `BIGCHERRY_ATTN_TS` vector cannot give three ranks KV in **one** QSA layer: `1,1,1` with `BIGCHERRY_ATTN_ROTATE=0` yields `(0,1,1)` head counts. Rotation can redistribute the active pair *between* layers, subject to effective-layer ordering. Keep validated production `1,1,0; rotate=0`; QFP07 owns physical-layout oracle and bounded static ABBA before any sequence-shard design. The existing QSA global top-k/mask and KV cache are not sequence-sharded by this patch. Host fixture passed; no new GPU performance claim.
