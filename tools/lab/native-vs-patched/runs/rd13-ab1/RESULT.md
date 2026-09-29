# rd13 (1206) vs control, Qwen3.8-27B-Q8_0, dual 7900 XTX, MTP n_max=4

Single variable: 1206_rd13_mul_mat_add_view_fusion (experiment rd13-only). 4 order-balanced pairs, sigint shutdown.
MTP acceptance identical between arms (work-equivalent).

| metric | delta |
|---|---|
| pp1024 | -0.03% |
| pp4096 | -0.03% |
| tg512 | +0.02% |
| tg2048 | +0.06% |

Flat: no measurable effect on this lane. Activation of the fusion path was not separately confirmed in this run, so
"no effect" cannot be distinguished from "path did not engage". Not a promotion candidate from this evidence.
