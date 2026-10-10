# 1330_qsa_mask_inplace

**Status:** evaluated
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

## Result: evaluated, ub1024 not adopted at the production context (2026-10-11)

The patch does what it was written for, and ubatch 1024 is still not usable at Flash-Next's production context.

- At ub512 the flag is neutral and the text is identical (2026-10-09). It only matters as the enabler for ub1024.
- With it, ub1024 loads at ctx 245760 and prefill is about 10 to 13% faster than ub512 at every depth that runs:
  1,376.9 / 1,371.6 against 1,206.9 / 1,228.9 t/s at 80K (240K context), 1,393 to 1,411 against 1,215 to 1,308 at
  49K, 1,392.5 / 1,392.7 against 1,181.6 / 1,281.5 at 73K (runs q17x4).
- At ctx 245760 it fails once about 118K tokens are in the context: `ROCm error: an illegal memory access was
  encountered` during the chunk that starts at token 117,760 (runs q17r, c200; 98K and 196K depths).
- The same binary and flags are clean at ctx 196608 with 124K tokens (run q17x3: 1,364.0 / 1,363.0 against 1,147.7
  / 1,252.9 t/s) and at ctx 131072 with or without this patch (runs q17x1, q17x2: same text in both ub1024 arms).
- At ctx 245760 it is also clean with sparse flash attention off (run q17y2, but then slower than production:
  1,091 against 1,256 t/s) and with the deferred catch-up off (run q17y1: 1,178 / 1,197 against 1,194 / 1,248, no
  gain left).
- Card memory in use at load, failing arm against production (rocm-smi, run q17r): 25.06 against 24.40 GB on
  XTX 0, 25.41 against 24.75 GB on XTX 1, 31.71 against 31.34 GB on the R9700. An RX 7900 XTX has 25.75 GB, so
  ub1024 leaves about 340 MB on XTX 1 where production leaves about 1 GB.

Reading: the configuration fits at load and runs out of card memory later, when the allocations made per call
(the sparse-attention index lists, the all-reduce's bf16 temporaries, pool growth) come on top of a longer context.
That reading fits every run above (smaller contexts are clean; taking either allocator of per-call memory out is
clean) but the faulting allocation itself was not identified: a run with kernels serialised hung at warm-up.

So: state `evaluated`. Not in the production recipe, not in the flashnext profile, and ub1024 is not adopted for
ctx 245760. It is a candidate again for a profile with a context of 196K or less, or once the per-call memory at
ub1024 is bounded (QFP47, the prefill scratch lease, is the plan item for that).
