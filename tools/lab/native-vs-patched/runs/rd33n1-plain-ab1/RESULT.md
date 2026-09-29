# rd33n1 (patch 1241, gate ne1==1) vs control, plain decode (no spec), tierL-qwen27b-q8, dual 7900 XTX -sm tensor

8 order-balanced pairs, bench set mtp-dual, `bigcherry ab-benchmark`, all 16 runs rc=0. Not performance-admitted
(execution attestation missing) -- exploratory evidence.

| metric | effect | 95% CI | pairs > 1.0 |
|---|---|---|---|
| pp1024 | +0.01% | [-0.16, +0.18] | 4/8 |
| pp4096 | -0.09% | [-0.18, -0.00] | 2/8 |
| tg512 | +4.37% | [+4.20, +4.54] | 8/8 |
| tg2048 | +4.38% | [+4.25, +4.59] | 8/8 |

Narrowing to ne1==1 retains the full plain-decode gain of the 1..8 gate (rd33-plain-ab1: +4.52% / +4.42%),
because single-token decode is the only shape that gate change affected outside MTP verify.
Combined with rd33n1-ab1 (MTP n_max=4: acceptance identical, tg +0.6%), rd33n1 is a strict improvement
over the 1..8 gate. Prefill cost is within 0.1%.

Does NOT attribute: correctness (reference-based contract campaign still required).
