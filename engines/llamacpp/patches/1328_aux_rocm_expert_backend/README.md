# 1328_aux_rocm_expert_backend

Rejected 2026-10-09 on a measured result at pin b11474 with its activation marker firing. Mechanism is in SUMMARY.md:
the routed experts of selected layers run on the RX 6900 XT through a plain ROCm backend.

## Evidence

Flash-Next UD-IQ4_XS, production profile (2x RX 7900 XTX + R9700 tensor split, MTP drafter on the RX 6900 XT,
flashnext features, look-ahead on, ctx 245760, f16 KV, ub512). One binary (production + 1328), chain `c152`,
A = production, B = experts of blocks 20 and 44 on the 6900 XT, order A B B A, two rounds.

| Depth | Prefill A (t/s) | Prefill B (t/s) | Decode A (t/s) | Decode B (t/s) |
|---|---|---|---|---|
| 8K | 1285.4 / 1283.9 / 1278.4 | 1190.0 / 1184.5 / 1185.2 / 1184.7 | 87.6 / 87.3 / 86.9 | 53.3 / 53.1 / 52.7 / 53.1 |
| 24K | 1309.3 / 1308.5 / 1311.1 / 1311.0 | 1209.6 / 1208.6 / 1206.9 / 1208.0 | 71.4 / 72.2 / 73.2 / 74.2 | 49.5 / 49.9 / 49.8 / 50.0 |
| 98K | 1263.9 / 1263.5 / 1259.9 / 1263.4 | 1120.0 / 1123.1 / 1121.3 / 1121.2 | 70.7 / 71.6 / 71.1 / 71.5 | 43.8 / 43.0 / 43.8 / 43.9 |

- Prefill -7.5% / -7.8% / -11.2% and decode -39% / -31% / -39% at 8K / 24K / 98K; every B run below every A run.
- Activation at 24K (A B B A): 0 / 7605 / 7533 / 0 marker lines.
- The greedy text differs between the arms at every depth.
- What it buys: about 0.49 GB of VRAM on the first XTX (24.41 GB used without, 23.92 GB with).

An earlier sweep with look-ahead off showed no prefill gain and -5.6% decode. The link to the 6900 XT is the cost:
the routed activations cross it twice per offloaded layer per step.

## What was kept

The one always-on edit this package carried, evicting only the stale split-state cache entry, is the cause of a
+3% prefill gain on its own and now lives in `1358_meta_split_cache_local_evict`.
