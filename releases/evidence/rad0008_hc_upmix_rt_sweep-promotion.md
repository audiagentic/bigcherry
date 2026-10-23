# rad0008_hc_upmix_rt_sweep promotion evidence

Promotion evidence, profile-evidence tier (QFP18 lightweight tier), radiance pin v1.9.0, 2026-10-23. Owner-approved
("if it helps and is not a burden then yes promote"). Promoted by hand: `bigcherry patch-promote` is llama.cpp-only
(MEN25).

Model: Qwen3.8-Flash-Next fp8-iq4r-moe (.rad), the only shape that reaches the PF2 up-mix (hc=4, n=2560, lr=320).

Mechanics: patch-lint clean; apply_patch on the pinned v1.9.0 libr4d/r4d_hc_bf16.hip applies, a second apply reports
already-applied; builds through the lane pipeline (RDNA4 libr4d, RDNA3 libr3) with no warnings added.
Activation: in-engine `radiance --profile-ops`, working layout (R9700 engine + two XTX helpers), one binary,
RAD_HC_UPMIX_RT=256 (the original launch) against the default 128, 7306-token prefill, 399 hc_read calls a run:
hc_read 218.6 / 214.6 ms against 207.6 / 212.4 ms (mean -3.0%, -6.6 ms; every RT=128 run below every RT=256 run).
Identity: byte-identical output at every size tested: r4d_selftest PF2 vs decode-tile kernels on the R9700, r3_selftest
on the XTX, rows 512-8192 including non-multiples of 128.
A/B: kernel level against the unpatched build, rotated rounds: RDNA4 R9700 -2.1% (M=2048) / -3.2% (M=4096), never slower
at rows 512-8192; RDNA3 XTX (libr3) -5.8% / -2.8%, complete separation. End to end in the working layout
(ab-benchmark, 4 pairs, one binary): pp4096 3351 -> 3355 tok/s, pp1024 3406 -> 3416, tg2048 152.3 -> 152.2: no
regression; the gain (~0.2-0.3% of prefill) is below run-to-run noise because hc_read is ~9% of prefill device time.
No-regression: decode cannot change (the PF2 form runs only for large prefill steps); end-to-end tg128/512/2048 unchanged.

Scope: profile-scoped. Only Qwen4Exp hyper-connection reads at the PF2 shape reach this launch.
Off switch: RAD_HC_UPMIX_RT=256 restores the original launch.

Plan items: RAD17, RAD18. Patch SUMMARY: engines/radiance/patches/rad0008_hc_upmix_rt_sweep/SUMMARY.md.
