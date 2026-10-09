# 1253_nro04_gfx1100_bf16_chunked_gdn

Plan `NRO04`; state `untested`; requires `1221_rd50_gdn_chunked_recurrence`.

RD50 supplies the earlier chunked-GDN infrastructure; 1253 adds the gfx1100 BF16 chunked path used by 1254. The implementation is live and is a required dependency of `1254_nro05_gdn_mtp_prefix_tail`; this README no longer describes it as predicate-only/incomplete scaffolding.

Runtime chunked-GDN opt-out is `GGML_CUDA_GDN_CHUNKED=0`. Keep BF16 operands with FP32 accumulation/state and preserve fail-closed fallback to the stock sequential path for unsupported shapes.

1254's MTP prefix/tail validation is separate: it must prove its own trace marker, full-vocabulary MTP correctness, snapshot/final-state semantics, work equivalence, and paired performance. Nothing here claims 1253 or 1254 is validated or signed off.
