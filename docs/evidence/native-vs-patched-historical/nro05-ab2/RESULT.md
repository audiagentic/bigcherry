# nro05 (1254) vs control, Qwen3.8-27B-Q8_0, dual 7900 XTX, MTP n_max=4

Single variable: 1254_nro05_gdn_mtp_prefix_tail (builds: bigcherry:stock:linux-multi, control vs --experiment nro05-only).
8 order-balanced pairs, sigint shutdown, MTP acceptance 0.90101 (1602/1778) identical in all 16 cells.
Activation: BIGCHERRY_PATCH_HIT patch=1254_nro05 path=gdn_mtp_prefix_bf16 seen on the nro05 build (traced run).

| metric | control | nro05 | delta | Mann-Whitney p |
|---|---|---|---|---|
| pp1024 | 992.65 | 1006.98 | +1.44% | 0.0002 |
| pp4096 | 1247.25 | 1273.68 | +2.12% | 0.0002 |
| tg512 | 88.86 | 88.73 | -0.14% | 0.20 |
| tg2048 | 101.70 | 101.77 | +0.07% | 0.46 |

Per-position means equal (no drift). Earlier 4-pair run (nro05-ab1) had a void control cell in pair 2 (acceptance 0.950, different work) and is superseded.
Not attributed: contract-grade correctness (MTP logprob parity), other archs, other models. run.json admission flag remains false (harness-level).
