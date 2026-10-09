# Testing — NRO04 gfx1100 BF16 chunked GDN

Static: RD50 dependency resolves; candidate predicate is gfx1100/RDNA3 only, `S_v==128`, scalar gate, K==1/prefill, and explicit opt-in; no live call is made in the draft.

Before kernel wiring, implement a gfx1100 WMMA fragment probe with identity, basis, structured and random 16x16/256-case matmuls. Confirm source-described interleaved accumulator rows, BF16 operand packing and FP32 accumulation.

Then compare one- and multi-chunk GDN output **and recurrent state** to FP32/CPU reference across gates, beta, sequence lengths and chunk boundaries. Quality tolerance must be fixed before timing. Non-gfx1100 builds must compile and never select the path.
