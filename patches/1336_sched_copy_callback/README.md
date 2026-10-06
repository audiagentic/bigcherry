# 1336_sched_copy_callback

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7). Mechanism is in SUMMARY.md. The
patch is a neutral enabler: it moves upstream's selective expert copy behind a public scheduler callback (the seam
MET01 needs for residency policy in user code) without changing what is computed. It is promoted because it is
proven transparent, not because it is faster. It only acts when routed experts are host-resident; the production
tensor split keeps every expert in VRAM and never reaches the callback.

## Evidence

Flash-Next UD-IQ4_XS, one GPU, routed experts in host memory, ctx 16384, f16 KV, no MTP
(`tools/lab/flash-next/moe-copy-ab.sh`). Arms on the patched build `b-moecopy-b11402g`: O = `BIGCHERRY_MOE_COPY=0`
(observation only, the scheduler copies every host weight whole), S = default (selective), R = no dense shortcut,
S2 = S again. Each arm serves, in one process: a short request, the same request again, a 4,702-token prompt, the
short request a third time. Reference: the same four arms on the production build without the patch
(`b-fadef-b11402e`), where the three environment variables do nothing.

| Card | Build | Prefill, 4,702 tokens (t/s) | Decode (t/s) |
|---|---|---|---|
| R9700 (gfx1201), 41 host layers | production | 90.1 / 90.5 / 90.3 / 90.5 | 18.3 / 18.4 / 17.3 / 18.4 |
| R9700 | patched, selective | 90.4 / 90.4 / 90.4 | 17.2 - 18.5 |
| R9700 | patched, observation | 59.7 | 18.2 |
| 7900 XTX (gfx1100), 45 host layers | production | 144.7 / 146.8 / 146.6 / 146.9 | 15.4 - 16.1 |
| 7900 XTX | patched, selective | 146.9 / 147.0 / 146.9 | 15.7 - 15.9 |
| 7900 XTX | patched, observation | 103.7 | 15.9 |

- Identity: the greedy text of every request is the same in every arm of the patched build and in the production
  build, on both cards (one md5 per request and card), including the second and third repeat of the short request
  in one process and the request after the workload shift.
- Activation: the exit counters (`BIGCHERRY_PATCH_TRACE`) show the callback called 1,230 times per arm on the R9700
  and 1,350 on the XTX; the selective arms copied 298.9 GB of 503.3 GB (R9700) and 324.9 GB of 551.0 GB (XTX) of
  eligible expert weights; the observation arm selected nothing and its prefill is 34% (R9700) and 29% (XTX) slower,
  so the callback is the path that delivers the selective copy.
- The dense shortcut (`BIGCHERRY_MOE_COPY_DENSE_PCT`, default 90) did not trigger at ub 512 on these prompts
  (59% of experts used per layer); arms S and R are equal. It is inert here and stays as a guarded default.
- Decode is not on this path: with host experts the scheduler offloads the expert matmul to the GPU only for prompt
  batches, single-token decode runs those experts on the CPU.
- Mechanics: offline tests (`tools/tests/patch/test_1336_sched_copy_callback.py`): applies alone and with 1326 in
  either order to the same result, idempotent, fails closed when the upstream block changes; patch-lint clean.

The first hardware attempt (`moecopy-b11402g-r9700`) is withdrawn: the script selected the card twice, no device was
left and the server ran on the CPU.

## Native llama.cpp comparison

Native llama.cpp b11402 already performs this selective copy inside the scheduler; the patch relocates it and the
table above shows the relocated form equal to the production build that still has upstream's embedded form (same
text, same speed). There is therefore no gain or loss against native from this patch by itself. Its purpose is the
callback seam for later residency work, and it becomes redundant when the pin reaches a release that contains
upstream #29943.
