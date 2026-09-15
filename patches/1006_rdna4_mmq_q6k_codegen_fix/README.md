# 1006 RDNA4 MMQ Q6_K codegen fix

Split from `1000_rdna4_mmq_q2k_q6k_fix` (rejected 2026-09-16, PA35) after
real hardware evidence separated the two independent edits it contained:
Q6_K genuinely improved (1.365x, CI95 1.362x-1.367x), Q2_K genuinely
regressed (0.959x, CI95 0.954x-0.963x). See
`patches/1000_rdna4_mmq_q2k_q6k_fix/SUMMARY.md` for the full disposition
record.

This patch carries only the Q6_K float-promotion edit. `state =
"untested"`: the combined-package measurement above is not reused as
promotion evidence for this different composition identity. A fresh
Q6-only exact-shape hardware A/B is required before promotion.
