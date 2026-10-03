# gp11 (1245) vs control at MTP n_max=5 (verify width 6), Qwen3.8-27B-Q8_0, dual 7900 XTX

Single variable: 1245_gp11_mmvq_fusion_ncols_gate. Config: server-ab-gp11-n5.json (only change from server-ab-gp11.json: --spec-draft-n-max 5).
4 order-balanced pairs, sigint shutdown. MTP acceptance 0.86705 in all 8 cells (work-equivalent).

| metric | effect | 95% CI |
|---|---|---|
| pp1024 | +0.05% | -0.15 .. +0.22 |
| pp4096 | +0.03% | -0.12 .. +0.18 |
| tg512 | -6.36% | -6.54 .. -6.23 |
| tg2048 | -6.43% | -6.58 .. -6.30 |

Regression: at the width where 1245's gate opens (ncols=6) decode is 6.4% slower with tight intervals, identical acceptance.
Together with gp11-ab1 (n_max=4, verify width 5, gate closed, flat) this shows the gate is width-specific and that the widened fused kernel
is slower on this lane (consistent with the register-pressure/scratch risk in the patch docstring; VGPR/scratch not inspected).
Activation still has no trace marker; the width-dependent effect is strong indirect evidence, not proof.
Verdict: not a winner; do not promote. Candidate for rejection unless a spill fix is pursued.
