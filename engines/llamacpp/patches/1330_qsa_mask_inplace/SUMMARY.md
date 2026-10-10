# 1330_qsa_mask_inplace

**Status:** untested
**Plan item:** QFP17

## What it does

With `BIGCHERRY_QSA_MASK_INPLACE=1`, the dense fallback in `build_qsa_sel` adds the causal `kq_mask` into the
QSA selection mask in place (`ggml_add_inplace`) instead of allocating a second dense `[n_kv, T]` F16 tensor.
For prompt batches with more than 8 tokens it pads the backing mask rows to 256 elements and passes the resulting
strided view directly to flash attention. Decode/MTP-sized batches (`n_tokens <= 8`) deliberately keep the exact
production allocation/add/reshape path; this avoids paying the observed ub512 decode cost for a memory optimization
needed only by prefill. `BIGCHERRY_QSA_MASK_INPLACE=2` inserts a contiguous copy before flash attention as a
diagnostic. Unset/`0` preserves the 1332 production path.

## Relationship to 1332

`1332_qsa_token_chunk` does not fully subsume this saving. With `BIGCHERRY_QSA_CHUNK` active, 1332 applies the
causal predicate while the selection is compact and, for batches with more than 8 tokens, returns I32 selection
indices before the dense fallback. `build_attn_qsa` then builds one final dense mask per token chunk and does no
dense causal-mask add. For decode/MTP-sized batches (`n_tokens <= 8`) and any other non-chunked fallback, 1332 keeps the original dense
path. 1330 now only replaces the add for `n_tokens > 8`; small batches retain the original out-of-place
`ggml_add(ctx0, sel, kq_mask)`.

1330 is therefore re-anchored after 1332 and requires it. 1332 already supplies the relaxed flash-attention
row-stride assertion; 1330 only changes the fallback mask construction/add/pass-through.

## Recheck

Build `deploy-v6-plus-1330` once and A/B the same binary with `BIGCHERRY_QSA_MASK_INPLACE=0` versus `1`.
Measure decode t/s, `BIGCHERRY_ALLOC_PEAK`/1331 peak live MiB (plus allocator buffer size), and greedy output
identity. Mode 2 remains available only to isolate a strided-mask read if mode 1 changes output.


## b11474 promotion candidate (2026-10-09)

Current-main Brutus recheck, Flash-Next, 240K context, ~80K fill: with
`BIGCHERRY_QSA_MASK_INPLACE=1`, ub1024 fits where production does not. Prefill was 1376.9 / 1371.6 t/s at
ub1024 versus production ub512 1206.9 / 1228.9 t/s (~+13%); decode was 66.1 / 66.3 versus 64.4 / 64.5 t/s.
At ub512/24K, one env A/B showed 75.7 / 76.3 t/s off versus 73.4 on, motivating the >8-token gate above.
Greedy text was identical.

Memory mechanism: at 240K/ub1024 the avoided F16 `[n_kv, T]` ADD result is ~480 MiB per Meta rank
(about 240 MiB at ub512). The prior peak attribution was ~1748 MiB/rank = 480 MiB kq_mask input + 484 MiB
mask_all + 480 MiB ADD result + 120 MiB kpool + ~180 MiB other. Removing the ADD result lowers the theoretical
live peak to ~1268 MiB/rank; 256-row padding adds only ~0.5 MiB at this shape. The observed ub1024 fit is the
mechanism proof.

Final 245760-context confirmation is pending. State remains **untested** until that result is posted.
