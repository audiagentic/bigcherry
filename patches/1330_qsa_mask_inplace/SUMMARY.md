# 1330_qsa_mask_inplace

**Status:** untested
**Plan item:** QFP17

## What it does

With `BIGCHERRY_QSA_MASK_INPLACE=1`, the dense fallback in `build_qsa_sel` adds the causal `kq_mask` into the
QSA selection mask in place (`ggml_add_inplace`) instead of allocating a second dense `[n_kv, T]` F16 tensor.
It pads the backing mask rows to 256 elements and passes the resulting strided view directly to flash attention.
`BIGCHERRY_QSA_MASK_INPLACE=2` inserts a contiguous copy before flash attention as a diagnostic. Unset/`0`
preserves the 1332 production path.

## Relationship to 1332

`1332_qsa_token_chunk` does not fully subsume this saving. With `BIGCHERRY_QSA_CHUNK` active, 1332 applies the
causal predicate while the selection is compact and, for batches with more than 8 tokens, returns I32 selection
indices before the dense fallback. `build_attn_qsa` then builds one final dense mask per token chunk and does no
dense causal-mask add. For decode/MTP-sized batches (`n_tokens <= 8`) and any other non-chunked fallback, 1332
keeps the original dense path, including the out-of-place `ggml_add(ctx0, sel, kq_mask)`.

1330 is therefore re-anchored after 1332 and requires it. 1332 already supplies the relaxed flash-attention
row-stride assertion; 1330 only changes the fallback mask construction/add/pass-through.

## Recheck

Build `deploy-v6-plus-1330` once and A/B the same binary with `BIGCHERRY_QSA_MASK_INPLACE=0` versus `1`.
Measure decode t/s, `BIGCHERRY_ALLOC_PEAK`/1331 peak live MiB (plus allocator buffer size), and greedy output
identity. Mode 2 remains available only to isolate a strided-mask read if mode 1 changes output.
