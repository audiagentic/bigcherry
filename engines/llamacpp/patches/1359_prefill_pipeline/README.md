# 1359_prefill_pipeline

Promoted 2026-10-11 (record below). Mechanism is in SUMMARY.md: with an MTP drafter, prompt batch k+1 is submitted before the drafter
hook collects batch k's hidden states, which it then takes through a backend event that waits for batch k only.
On by default; `BIGCHERRY_PREFILL_PIPELINE=0` turns it off.

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

## Promotion record

Promotion evidence, profile-evidence tier (QFP18), pin b11474, 2026-10-11.

Model: Qwen3.8-Flash-Next UD-IQ4_XS (production profile: 2x RX 7900 XTX + R9700 tensor split, MTP drafter on the RX 6900 XT, ctx 245760, f16 KV, ub512)
Model: Qwen3.8-27B Q8_0 (two RX 7900 XTX, tensor split, built-in MTP)

Mechanics: offline tests for 1359 pass (3 tests: applies after the production set, idempotent, fails closed on a drifted anchor), patch-lint adds no warning, composition with the production set clean (CI offline-patch and compile-only on PR #135).
Activation: BIGCHERRY_PATCH_HIT patch=1359_prefill_pipeline with the flag on only (run pipe1-d24576: off arm 0, on arm 1).
Identity: greedy text identical to the flag off in every run: 8K, 24K and 98K (two ABBAs each), twelve runs with the flag on at 98K against one baseline (run pipe3: one md5, b301a49a), prompts of 418 and 1,186 tokens (under one and under three batches, run pipe5), four different requests at 24K (run dacc-p1359-*). The mechanism is a reordering of host work; the order of work on each card's stream is unchanged.
A/B: one binary (b-pipe1 = production + 1359), BIGCHERRY_PREFILL_PIPELINE=1 against default (off). Prefill +6.0% at 8K, +7.4% to +10.7% at 24K, +10.0% at 98K, every on-run above every off-run at each depth; four requests at 24K, ABBA each: +15.9%, +11.5%, +11.1%, +12.7% (pooled +12.8%). Decode and draft acceptance unchanged.
No-regression: Qwen3.8-27B on two RX 7900 XTX, same binary, flag on against off (prod27b-ab.sh, run nr27-1359): text identical at 10K and 32K, prefill and decode unchanged. The pipeline does not engage there (the built-in MTP shares the target's memory, so 1348's deferred path is off).

Scope: profile-scoped. The flag stays default off; the patch engages only with an MTP drafter in its own context and BIGCHERRY_MTP_DEFERRED_CATCHUP on.

## Flash-Next, one binary, pipeline on against off

`tools/lab/flash-next/queue-env-ab.sh`, A = default (off), B = BIGCHERRY_PREFILL_PIPELINE=1. Prefill t/s:

| Depth | Off | On | Gain |
|---|---|---|---|
| 418 tokens (run pipe5) | 357.5 / 356.0 | 367.8 / 367.2 | +3.0% |
| 1,186 tokens (run pipe5) | 669.8 / 676.3 | 703.1 / 712.3 | +5.2% |
| 8K (run pipe2) | 1306.7 / 1309.7 | 1380.2 / 1393.4 | +6.0% |
| 24K (run pipe1) | 1261.3 / 1328.4 | 1437.5 / 1429.1 | +7.4% against the higher off-run, +10.7% on the means |
| 98K (run pipe2) | 1226.4 / 1275.6 | 1377.7 / 1373.5 | +10.0% |

Decode 79.9-80.8 t/s at 8K, 89.3-90.0 at 24K and 65.7-66.7 at 98K in both arms; acceptance 329-330/540-545, 353-355/467-473 and 335-336/523-526.

Four requests at 24K (`queue-decode-acceptance.sh`, SWITCHES p1359): prefill off / on 1231.2 / 1426.9, 1265.2 / 1410.8, 1274.8 / 1416.3, 1265.2 / 1425.7; decode pooled 74.2 against 74.7 t/s; acceptance 60.6% against 61.0%; same text in all four.

## What the cards do (kernel trace, 24K, per target card)

`tools/lab/flash-next/kernel-gap-stats.py` over the prefill window, production (run dp1) against the pipeline on (run pipe4):

| | Off | On |
|---|---|---|
| gaps of 10 ms or longer | 81, 2.78 s | 13, 0.81 s |
| gaps of 2-10 ms | 190, 0.81 s | 293, 1.28 s |
| idle share of the window | 17.4% | 12.9% |

The once-a-batch gap is gone. The upper bound for this mechanism is the same prefill with no drafter, +14.6% (run nomtp1).

## Host threading condition

None applies: the patch adds no thread. Shared state between calls is the two pinned buffers and their events, written by `llama_context::decode` and read by the hook on the same thread; a buffer is not reused until its events have been waited for.

## Not covered

A request cancelled in mid-prompt and a context shift have no hardware run. Both go through `reset_deferred`, which drops the outstanding batch by the rule 1348 uses for a pending snapshot. 1346's prompt timing counts the synchronous path only, so its chunk counters read zero with the pipeline on.
