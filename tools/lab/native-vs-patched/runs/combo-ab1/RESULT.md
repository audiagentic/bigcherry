# dense-decode-combo (1241+1206+1245) vs control, Qwen3.8-27B-Q8_0, dual 7900 XTX, MTP n_max=4

VOID for throughput attribution: work-equivalence failed. Control acceptance 0.90101 (1602/1778) in all 8 cells;
combo acceptance 0.95580 (1622/1697) in all 8 cells -> the combo generates different tokens (numerics diverge under greedy),
so tg deltas compare different work.

Raw (8 pairs, order-balanced, no position drift): pp1024 -0.25%, pp4096 -0.14%, tg512 +1.80%, tg2048 -5.48% (all p<=0.03).
Not attributed: which of 1241/1206/1245 changes the output; 1245 has no trace marker (composition verified from guard strings in source only).
Next: single-patch arms (rd33-only, rd13-only, gp11-fusion-ncols) to find the divergence source.
