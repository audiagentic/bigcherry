# ENG05 step 2: MTP depth sweep on flashnext-4gpu

Measured 2026-11-20 07:04-08:05 AEDT (2026-11-19 20:04-21:05 UTC), under the `gpu` lease, 20-CPU host.
Build: main as of #1210's base (plan with the current serving and radiance), `engine.num_speculative_tokens` 2, 3, 4 with
`--set` through `FLASHNEXT_PARITY_SETS`; launch order 3 2 4 4 2 3 (two launches per depth, mirrored to cancel drift).

## Method

`tools/lab/radiance/flashnext-parity.sh measure --launches 1` per point with three changes, none on the decode path:

- Placement policy off in every arm. The calibration store is keyed by a hash of the manifest and the depth is part of
  it, so only the stock depth 3 has a record (the same finding as the combine-thread sweep).
- The cold-prefill and document-prefill rounds are skipped.
- Decode at 1 and 6 streams is run for three prompt seeds (7, 21, 33), two reps each; then sess8 (8 sessions) and the
  256K context (seeds 7 and 8, two rounds each, the first cold).

Draft acceptance is a property of the prompt, so it differs by seed; each metric below is the mean over the records of
one depth and load. The 451 s policy warmup does not apply with the policy off.

## Result (mean over records; decode tok/s is aggregate across the streams)

| load | depth | ms/step | tok/s | acceptance | tokens/step |
|---|---|---|---|---|---|
| 1 stream | 2 | 26.48 | 91.6 | 0.855 | 2.71 |
| | 3 | 25.79 | 107.2 | 0.765 | 3.30 |
| | 4 | 27.50 | **110.6** | 0.666 | 3.66 |
| 6 streams | 2 | 45.01 | 225.9 | 0.764 | 14.30 |
| | 3 | 49.00 | **233.2** | 0.673 | 16.17 |
| | 4 | 54.77 | 230.7 | 0.585 | 17.89 |
| 8 sessions | 2 | 40.77 | 98.9 | 0.820 | 19.93 |
| | 3 | 44.31 | **108.5** | 0.811 | 25.98 |
| | 4 | 49.33 | 103.6 | 0.656 | 29.22 |
| 256K, 1 stream | 2 | 27.01 | 40.7 | 0.683 | 2.37 |
| | 3 | 29.56 | 42.4 | 0.595 | 2.79 |
| | 4 | 31.60 | **46.5** | 0.575 | 3.30 |

Spread between the two launches of a depth (tok/s): 1 stream 0.0 (4), 0.1 (3), 8.1 (2); 256K 0.0 (4), 0.2 (3), 0.0 (2);
6 streams 1.7 (4), 6.6 (3), 5.8 (2); sess8 0.7 (3), 2.1 (2), 9.4 (4: 98.9 and 108.3).

## Reading

- Depth 2 loses everywhere (-15% at 1 stream, -3% at 6 streams, -9% at 8 sessions, -4% at 256K): drop it.
- Depth 4 wins where the step is cheap: +3.1% tok/s at 1 stream (the three prompt seeds give +1.0%, +4.8%, +3.1%, both
  launches identical) and +9.7% at 256K (46.5 against 42.4, spread 0.1). Its step is 1.7 ms longer at 1 stream and 2 ms
  at 256K, paid for by 0.36 and 0.5 more tokens a step.
- Depth 3 wins or ties under load: at 6 streams depth 4 is -1.1% (step 54.8 against 49.0 ms for 1.7 more tokens), at 8
  sessions it is -4.5% on the mean but one launch ties (108.3 against 108.5) and the other is 9% low, so call it a tie at best.
- So the best depth does differ by load: 4 at one stream and at long context, 3 from six streams up.

## What a load-dependent depth needs

The engine already has it: `--draft-window adaptive` (core/sched/draft.h), where `--num-speculative-tokens` is the
ceiling and the depth per step follows measured acceptance and step cost, with no threshold chosen by hand. The
engine's own comment records it losing on a busy Flash-Next TP2 at ceiling 5 (-11.9% at 8 sequences), so it has to be
measured here, at ceiling 4, before it can be trusted on this layout. The serving layer has no key for it:
`draft_window` exists in the manifest contract (`/spec/processes/engine/tuning/draft_window`) but the resolver, the
manifest writer and the engine arguments do not set it, so every launch is `fixed`. Next slice: an `engine.draft_window`
setting (fixed or adaptive, fixed by default), then adaptive at ceilings 4 and 5 against fixed 3 and fixed 4 on this
sweep's loads, policy off and (once calibrated) on.

## Adaptive draft window and a fixed-3 control (same day, 07:59-08:45 AEDT)

`tuning.draft_window` is already a resolver key (the generic tuning options derive from the manifest contract), so no
knob was needed: `--set tuning.draft_window=adaptive --set engine.num_speculative_tokens=N` puts `--draft-window
adaptive` and the ceiling N on the engine (engine log: `draft depth: adaptive, 1..N`). Two launches each at ceilings 4
and 5, plus one more fixed-3 launch as the drift control, same harness and loads, policy off. (The earlier version of
this page said the key did not exist; that was wrong.)

| arm | 1 stream tok/s | 6 streams | 256K | 1 stream ms/step | acceptance (1 stream) |
|---|---|---|---|---|---|
| fixed 3 (two sessions) | 107.2, 104.4 | 233.2, 227.4 | 42.4, 42.5 | 25.8, 25.9 | 0.765 |
| fixed 4 | 110.6 | 230.7 | 46.5 | 27.5 | 0.666 |
| adaptive, ceiling 4 | 110.4 | 231.1 | 46.5 | 27.6 | 0.666 |
| adaptive, ceiling 5 | 101.2 | 210.3 | 42.5 | 29.8 | 0.508 |

- Adaptive at ceiling 4 is fixed 4: same step time and the same acceptance at 1 stream, 6 streams and 256K, so its
  back-off never engages on these loads, and it adds nothing. At ceiling 5 it is worse everywhere (the engine's own
  note about it losing under load holds here, and it also loses at 1 stream).
- Fixed 4 against fixed 3 (average of the two sessions, 105.8 at 1 stream): +4.5% at 1 stream, +9.5% at 256K, a tie at
  6 streams (230.7 against 230.3). The two fixed-3 sessions differ by up to 6 tok/s at 6 streams, which is also why the
  first page's "depth 4 loses at 6 streams" is inside the noise.
- sess8 is not a usable tok/s comparison here: it is the median of two reps and one of them is a cold 36-41 s prefill
  whenever the prefix cache missed (adaptive-4 launch 1: both reps cold, 6.7 tok/s aggregate; launch 4: second rep
  1.3 s, 99.6). Its decode ms/step is steadier: fixed 3 44.3-44.8, fixed 4 49.3, adaptive-4 48.2, adaptive-5 51.0.

## Decision

A fixed depth 4 is the lever on this layout (+4.5% one stream, +9.5% at 256K, no loss at 6 streams), but it is not
shippable as a default change yet: the live-policy calibration key hashes the depth, so a depth-4 launch has no
calibration record and runs with the policy inactive until one is calibrated (every arm above ran policy-off for
that reason). The default change therefore needs a depth-4 calibration record, then a policy-on A/B of 3 against 4 on
these loads (including prefill/TTFT, which must not regress). That is plan item ENG08; adaptive is closed.
