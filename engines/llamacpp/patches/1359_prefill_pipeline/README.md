# 1359_prefill_pipeline

Not promoted. Mechanism is in SUMMARY.md: with an MTP drafter, prompt batch k+1 is submitted before the drafter
hook collects batch k's hidden states, which it then takes through a backend event that waits for batch k only.
Off by default; `BIGCHERRY_PREFILL_PIPELINE=1` turns it on.

## Evidence

- Gate (Flash-Next 24K, production profile): each target card has nothing queued for about 34 ms a 512-token batch
  (kernel trace `dp1`, `tools/lab/flash-next/kernel-gap-stats.py`); the host's one wait a batch is the hook's
  `llama_get_embeddings_nextn` (synchronisation trace `ps1`); without a drafter prefill is 14.6% faster (`nomtp1`).
- Mechanism outside the engine: `tools/lab/hip-probes`, test `pipeline`, run `hp6`.

## Still to show

Listed in SUMMARY.md. A changed greedy text rejects the patch.

## Native llama.cpp baseline

Native llama.cpp b11474 calls the drafter hook synchronously after every prompt batch: it waits for the batch and
runs the drafter's catch-up before the next batch is built. `BIGCHERRY_MTP_DEFERRED_CATCHUP=0` restores exactly that
order (it is 1348's own off switch), so the native behaviour for the code this patch changes is an arm of the same
binary. The rest of the build is the BigCherry production baseline in every arm.

Flash-Next, 24K, build `b-pipe1` (production + 1359), ABBA each, prefill t/s:

| Arm | Runs | Against native order |
|---|---|---|
| native order (`BIGCHERRY_MTP_DEFERRED_CATCHUP=0`), run `pipe6` | 1231.2 / 1246.3 | |
| 1348 deferred catch-up, the production default, run `pipe1` | 1261.3 / 1328.4 | +4.5% on the means (+7.7% in the other 24K pairs of the day, about 1,335) |
| 1359 pipeline on, run `pipe6` | 1433.6 / 1432.8 | +15.7% |

Greedy text identical in all three arms (md5 a789c151), decode 89.5 to 90.6 t/s and acceptance 353-354 / 470-473 in
all. So the gain is against the native order as well as against BigCherry's own baseline: 1348 took part of the
drafter's cost out of the prompt path and this patch takes most of what was left.
