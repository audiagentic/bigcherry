---
id: RHP10
order: 10
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-31T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Long-context thrash experiment

## Description

Test whether expert residency changes explain the 290K-1M decode slowdown (decode 92-122 tok/s to ~290K, 74 at 580K, 50 at 1M). A falling decode rate does not by itself show thrashing: growing KV attention work and memory pressure compete as explanations. Prior evidence: a 6 GiB tail cut expert promotions 8,084 -> 5,767 and host-to-device traffic 51.3 -> 32.2 GiB over a short-context run with no speed change.

## Steps

1. Write the protocol, experiment or collector against the code as it is (radiance 1.14.0, remote-MoE protocol v19).
2. Land card-free tests first where the item has any logic; measure only with the gpu lease.

## Detailed Solution & Technical Design

## Code Samples & Guidance

## Files

## Validation

At 1M context with tails of 0 / 3 / 6 / 14 GiB, with the context filled to ~290K and ~580K: resident expert bytes, misses, mover activity, transfer volume, KV occupancy and decode/prefill throughput; the result decides RHP05-07.

## Effort & Risk

Depends on: none (needs the gpu lease and all four cards). Added by the 2026-10-31 GPT review of this plan.

## First long-context runs, 2026-10-31 (`rad23-longctx`, `rad23-longctx2`)

Flash-Next, `flashnext-2xtx-rdna2-tail` (4 GiB tail), `ctxsweep.py` over the documentation text; used context 330K and 688K (the 295K and 590K targets tokenise longer). Static heat is `perf51/heat.rank0.hot-first`; live heat is the snapshot of RHP08 inverted the same way. The 688K row of `rad23-longctx` follows a 330K request that stays cached; `rad23-longctx2` is the 688K request alone on a fresh server, three arms back to back.

| Run | Arm | Prefill tok/s | Decode tok/s | Promotions | Host-to-GPU GiB |
|---|---|---|---|---|---|
| after 330K | XTX 22,528, static | 2413 / 1392 | 113 / 69.8 | 11365 | 2480 |
| after 330K | XTX 23,500, live | 2507 / 1492 | 110 / 87.7 | 10001 | 2126 |
| 688K alone | XTX 22,528, static | 1833 | 77.9 | 8929 | 1128 |
| 688K alone | XTX 23,500, static | 1893 | 74.8 | 8002 | 993 |
| 688K alone | XTX 23,500, live | 1944 | 98.5 | 8067 | 970 |

(Prefill/decode pairs in the first two rows are 330K / 688K.) The extra XTX budget alone cuts host-to-GPU traffic by 12% and does not move decode; adding the live heat on top gives +26% decode at 688K in both settings (87.7 against 69.8, 98.5 against 77.9) and +4 to 7% prefill. Prefill at 688K is 1392 to 1944 tok/s depending on what is already cached, so only runs set up the same way compare. Decode pairs spread widely inside a run (69 to 86), so the second fact to keep is one run per arm in the second table.

Caveat that blocks a conclusion: the live heat was recorded on the same documentation text the sweep reads, so it is fitted to the test. The held-out test is heat from one corpus against a sweep over another.

## Held-out corpus, 2026-10-31 (`rad23-holdout`)

The same server and tail as above, a sweep over C++/HIP source (the radiance source tree, 29 M characters, one request of 675,534 tokens, fresh server per arm). Heat from the documentation text of the arms above is held out; heat from this corpus is fitted to it. The first baseline run recorded the route histogram (the engine counts every route in VRAM), which slows it, so the control is the plain `base2` run; its traffic figures are identical to the histogram run's.

| Arm | Prefill tok/s | Decode tok/s | Promotions | Host-to-GPU GiB |
|---|---|---|---|---|
| baseline: XTX 22,528, static heat (`base2`) | 1450 | 77.5 | 8908 | 1088 |
| baseline recording the histogram | 1412 | 71.2 | 8908 | 1088 |
| XTX 23,500, static heat | 1497 | 80.2 | 7980 | 959 |
| XTX 23,500, heat from the documentation text (held out) | 1946 | 77.4 | 8374 | 938 |
| XTX 23,500, heat from this corpus (fitted) | 1963 | 81.2 | 8078 | 983 |

Prefill is +34% with heat the sweep never saw and +35% with fitted heat, against +3% for the larger budget alone; decode is level here (the spread inside a run is 70 to 93), where on the documentation corpus the same change gave decode +26% and prefill +4 to 7%. So the larger XTX budget cuts host-to-GPU traffic by 12 to 14% on both corpora and the workload-recorded heat moves speed at long context, by prefill on one corpus and decode on the other. Not yet separated: how much comes from the XTX sets and how much from which experts end up in the 6900 XT's tail (the heat file decides both). One run per arm; the direction repeats on two corpora and across two orderings, the size does not.

Method to adopt: record with `R3_MOE_WORKER_HISTOGRAM` and `R3_MOE_WORKER_HISTOGRAM_PERIOD_S` on the real workload (the counting has a cost, so record, do not serve with it), invert (`max - count`), restart with `--heat` and `--budget gfx1100=23500`.

## Quality of the recorded-heat restart, and which part pays, 2026-10-31 (`rad23-kld-heat-10311458`, `rad23-kld-heat2-*`, `rad23-holdout`)

The quality harness of RAD23 (8,240 positions against the 2026-10-23 exact reference), a two-helper control beside the arms, every arm with the 6900 XT as a 4 GiB tail. Heat is the static profile or the snapshot recorded on the documentation text (the quality positions are documentation text too, so the live-heat arms are fitted to them).

| XTX budget MiB | Heat | KLD mean | median | p99 | p99.9 | top-1 % | PPL | Gate |
|---|---|---|---|---|---|---|---|---|
| 22,528, two helpers (control) | static | 0.05518 | 0.01419 | 0.5557 | 1.6424 | 89.745 | 9.0651 | |
| 22,528 | static | 0.06205 | 0.01629 | 0.6704 | 2.2477 | 89.029 | 9.1025 | fails p99.9 (+0.61, limit +0.5) |
| 22,528 | live | 0.06264 | 0.01564 | 0.6382 | 1.9263 | 88.920 | 9.0628 | passes (p99.9 +0.28) |
| 23,500 | static | 0.06573 | 0.01602 | 0.7223 | 2.0600 | 88.701 | 9.0381 | fails top-1 (-1.04 points, limit 1.0) |
| 23,500 | live | 0.06602 | 0.01581 | 0.6886 | 2.7905 | 89.090 | 9.1164 | fails p99.9 (+1.15) |

The mean rises with the XTX budget (about +0.0035 for the extra 2 GiB, about 850 experts that now take the helper's path instead of the engine's own) and not with the heat file. The tail of the distribution is noisy: p99.9 rests on about eight positions and swings between 1.93 and 2.79 across arms that differ only in placement, so one run of it decides nothing; the mean and top-1 move less. The 4 GiB tail with static heat already breaches p99.9 at the normal budget, as the 6 GiB tail did in the earlier sizes run.

Speed on the held-out code corpus (675,534 tokens used, fresh server per arm): XTX 22,528 with the live heat gives prefill 1883 tok/s (+30% on the plain 1450) and decode 68.5 (inside the spread of the 77.5 baseline), host-to-GPU 1068 GiB (-2%). So the prefill gain is the heat file and not the budget (budget alone: +3%), and the variant that carries it, live heat at the normal budget, is the one that passes the quality gate. The larger budget buys traffic (-12 to 14%) at a measurable quality cost and no further speed; it is not adopted.

Open: why the helper's path costs KLD per expert moved (the loss scales with the experts held by helpers, which says the helper arithmetic differs from the engine's in a way that matters), and whether a heat file recorded on a different workload from the one served keeps its gain (held-out prefill says yes, once).

## Two concurrent streams, 2026-10-31 (`rad23-streams2`)

The launcher's two sequences, both busy: two requests at once, each a different slice of the corpus (about 65,000 or 8,000 prompt tokens, 400 new tokens each, `ignore_eos`), three repetitions per cell on one server, the overall tokens a second counting prompt plus new tokens. XTX 22,528 MiB and the 4 GiB tail in both arms; heat the static profile or the snapshot recorded on the documentation text. The rate falls with the repetition because the cache fills (rep 0 / 1 / 2 below).

| Corpus, prompt per stream | Static heat | Recorded heat |
|---|---|---|
| documentation, 65K | 2622 / 2241 / 1892 | 2625 / 2228 / 1934 |
| documentation, 8K | 1265 / 1166 / 1251 | 1287 / 1187 / 1257 |
| C++/HIP source, 65K | 1412 / 1232 / 1076 | 1544 / 1323 / 1128 |
| C++/HIP source, 8K | 772 / 833 / 798 | 768 / 791 / 779 |

Recorded heat is level on the corpus it was recorded on and at 8K, and +5 to 9% on the other corpus at 65K a stream (about 130K tokens used in all). So the +30% of the single 675K request belongs to the regime where the cache thrashes; at two streams of 65K the heat file is worth a few percent at most. Aggregate rate with two streams of 65K prompts is 1.9K to 2.6K tok/s on documentation text.

## The four-way cell on the held-out corpus, and the repeat spread, 2026-10-31 (`rad23-holdout`)

One request (675,534 tokens used), one fresh server per arm, XTX 22,528 MiB:

| Tail | Heat | Prefill tok/s | Decode tok/s | Promotions | Host-to-GPU GiB |
|---|---|---|---|---|---|
| 6900 XT 4 GiB | static | 1450 | 77.5 | 8908 | 1088 |
| 6900 XT 4 GiB | recorded | 1883 | 68.5 | 9203 | 1068 |
| none | static | 1792 | 57.1 | 11214 | 1373 |
| none | recorded | 1843 | 83.2 | 10798 | 1241 |

Spread, measured: the tail + static arm run three times (1450, 1448, 1468 prefill; 77.5, 77.4, 77.6 decode; promotions and traffic identical to the byte). For a fixed request the system is deterministic to about 1.4% in prefill and 0.3% in decode, so the differences in the table are real for this request. They are not monotone, which is the finding: the 6900 XT tail costs 19% of prefill with the static heat (1450 against 1792) and costs none with the recorded heat (1883 against 1843); it adds 36% decode with the static heat (77.5 against 57.1) and loses 18% with the recorded one (68.5 against 83.2); recorded heat is +3% prefill and +46% decode without the tail, +30% prefill and -12% decode with it. The best decode cell is no tail with recorded heat (83.2) and the best prefill cell is the tail with recorded heat (1883); no cell wins both. Host-to-GPU traffic falls with both the tail and the recorded heat in every pairing.

Two caveats that keep this from a recommendation. It is one request on one corpus (a different prompt will not repeat these numbers; the earlier documentation-corpus run and the two-stream runs gave smaller and differently ordered effects), and the interaction means the sign of the tail's effect depends on which experts the heat file puts where, which is why single-factor claims (the tail helps, the heat helps) fail. The earlier "+30% prefill from recorded heat" holds only with the tail and costs decode there; without the tail the heat file's gain is decode.

Next: the same four cells on a second request and corpus, and a per-layer trace of the helper waits in the two cells that differ most (tail + static, no tail + static) to see which helper the engine waits for.
