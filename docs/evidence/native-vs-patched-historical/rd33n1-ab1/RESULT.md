# rd33n1 (patch 1241, gate ne1==1) vs control, MTP n_max=4, tierL-qwen27b-q8, dual 7900 XTX -sm tensor

8 order-balanced pairs, bench set mtp-dual, `bigcherry ab-benchmark`. Not performance-admitted
(execution attestation missing) -- exploratory/wiring evidence only.

| metric | effect | 95% CI | pairs > 1.0 |
|---|---|---|---|
| pp1024 | -0.02% | [-0.28, +0.23] | 4/8 |
| pp4096 | +0.14% | [-0.05, +0.46] | 5/8 |
| tg512 | +0.58% | [+0.34, +0.83] | 8/8 |
| tg2048 | +0.68% | [+0.41, +0.92] | 7/8 (one pair 0.9999) |

Work equivalence: draft acceptance 0.90101 (1602/1778, mean len 4.60) identical in all 16 runs.
The widened 1..8 gate (rd33-ab2) changed acceptance (0.9558 vs 0.9010) and lost tg2048 -5.65%;
narrowing to ne1==1 removes both. Activation (ncols=1 only) shown in ../rd33n1-activation.

Does NOT attribute: correctness (needs the reference-based contract), plain-decode magnitude for rd33n1
(rd33 with 1..8 gave +4.4% plain; rd33n1 plain A/B still to run).
