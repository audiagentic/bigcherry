# gp11 (1245) vs control, Qwen3.8-27B-Q8_0, dual 7900 XTX, MTP n_max=4

Single variable: 1245_gp11_mmvq_fusion_ncols_gate (experiment gp11-fusion-ncols). 4 order-balanced pairs, sigint shutdown.
MTP acceptance 0.90101 in all 8 cells (work-equivalent).

| metric | effect | 95% CI (bootstrap, paired) |
|---|---|---|
| pp1024 | +0.02% | -0.13 .. +0.17 |
| pp4096 | +0.06% | -0.03 .. +0.14 |
| tg512 | -0.31% | -0.63 .. +0.03 |
| tg2048 | +0.01% | -0.41 .. +0.42 |

Flat: every interval spans zero. Not a winner on this lane.
Activation of the gated fusion path is NOT proven (1245 has no trace marker), so "no effect" cannot be separated from
"path did not engage for 27B Q8_0 MTP decode". Not attributed: any effect on other models/lanes.
