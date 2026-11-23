# The IQ8R int8 GEMM row on each card (rad-kbench)

`rad-kbench -m ltx-2.5-iq8r.rad --op gemm_nt_q --max-tok 33833 --timing-only --max-cases 1`, one visible card per run
(device timestamps around back-to-back launches, sustained warm-up, weights read from memory). The row on both cards is
`gemm_iq8r_tiled`: libr4d's source on the gfx1201 card, libr3's gfx11 build of it on the gfx1100 card.
`gemm_iq8r_tiled.md` pairs the two cards' shapes (M is each card's own tile rounding); TOPS = 2·M·N·K / time.
