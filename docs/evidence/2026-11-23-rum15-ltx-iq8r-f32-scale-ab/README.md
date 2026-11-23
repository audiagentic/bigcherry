# RUM15: the F32-scale IQ8R layout on the LTX GEMM shapes

`rad-kbench -m ltx-2.5-iq8r.rad --op gemm_nt_q --max-tok 33833 --timing-only --max-cases 1`, one card per run, stock
kernels against the diagnostic F32-scale build (rad0061 libr3 and libr4d, same libref), each arm under its own short
GPU lease, the same shapes on the same checkpoint. Every gemm_nt_q shape the LTX graph declares for the IQ8R rows is
timed (the large-M tiled row and the M <= 64 row), not only the Qwen fixtures the earlier RUM15 notes used.

`table.md` has the results. The large-M tiled row, which every DiT block runs at 2048 tokens and above, is unchanged
within noise on both cards (median -0.47% gfx1201, +0.26% gfx1100 at M >= 2048); the M <= 64 row's median moves under
1%, with a tail of single shapes up to +17-18%. The earlier Qwen-fixture figures (+8 to +25% on individual narrow
shapes) are therefore not representative of LTX's prefill-like path. The memory cost stands: the folded F32 scales add
242.8 MiB for the whole model.
