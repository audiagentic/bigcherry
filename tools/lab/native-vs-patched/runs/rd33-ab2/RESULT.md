# rd33 (1241) vs control, MTP n_max=4, 8 order-balanced pairs (supersedes rd33-ab1)

Acceptance differs between arms in every cell: control 0.90101, rd33 0.95580 (deterministic). The arms generate different text, so
throughput is NOT a same-work comparison; read the numbers as an end-to-end effect of a patch that changes outputs, not as attributable kernel speed.

| metric | effect | 95% CI |
|---|---|---|
| pp1024 | -0.29% | -0.40 .. -0.16 |
| pp4096 | -0.14% | -0.20 .. -0.08 |
| tg512 | +1.49% | +1.28 .. +1.69 |
| tg2048 | -5.65% | -5.80 .. -5.51 |

Signs disagree across generation lengths (tg512 up, tg2048 down), consistent with content-dependent behaviour of different generated text.
Conclusion for the production MTP lane: rd33 shows no reliable win and a large tg2048 loss; its measured benefit is plain (non-speculative) decode
only (see ../rd33-divergence, +4.9%). Do not promote for the MTP lane. Whether it is worth promoting for non-MTP dual-XTX decode is pending
the order-balanced plain-decode A/B (rd33-plain-ab1).
