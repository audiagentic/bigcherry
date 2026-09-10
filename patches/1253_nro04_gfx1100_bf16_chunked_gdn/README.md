# 1253_nro04_gfx1100_bf16_chunked_gdn

Plan `NRO04`; state `untested`; requires `1221_rd50_gdn_chunked_recurrence`.

RD50 supplies the earlier chunked-GDN infrastructure but only selects RDNA3.5. Nasone's newer block adds a separately segregated gfx11 BF16/WMMA kernel. This package's first draft adds the runtime architecture/shape policy and an explicit opt-in environment gate while leaving live selection false until the first-generation WMMA fragment mapping is validated on gfx1100.

The eventual kernel must be folded into `gated_delta_net.cu` because BigCherry's anchored patcher cannot create upstream source files. It must keep BF16 operands and FP32 accumulation/state, and it must not share the gfx12 kernel implementation.

The draft is intentionally incomplete rather than shipping unverified WMMA code. See `TESTING.md` for the matrix-probe prerequisite.
