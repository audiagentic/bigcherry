# GP11 / HI158 -- replay-vs-control end-to-end bench

Plan items: HI158 (lazy native provider), GP11 (dispatch overhead).

## Question

Does a `replay` build carrying tuned winners actually beat a `control` build
carrying none, end-to-end on a real server workload? The tuner promotes winners
on microbenchmark evidence; nothing so far has checked that those winners
survive contact with the full graph.

Secondary: what does the dispatch framework itself cost, measured as
`bigcherry control` (dispatch layer compiled in, no winners) against
`llama-native` (no dispatch layer at all)? The target is zero.

## Inputs

- 27B model, dual RX 7900 XTX (gfx1100), `-sm tensor`, MTP speculative decode.
- Three pre-built binaries, located by build digest under
  `~/.cache/bigcherry/builds`:
  - `a55fa53d6c9c63e01115aa09847f77eb` -- llama-native (0 patches)
  - `df75a6d33c4d2d5342e567ca2a6b01ba` -- bigcherry control (no winners)
  - `e59994bc49764809b1b4b957d71e934d` -- bigcherry replay
- Replay arm additionally sets `GGML_HIP_DISPATCH_CACHE` to the tune campaign's
  `dispatch.cache` (campaign `b4d3cf708425`, 19 promoted winners).
- Measurement is the documented `bench/run_bench.py --bench-type server-bench`
  harness (see `docs/reference/testing/TEST.md`), never `llama-bench` -- MTP is
  invisible to `llama-bench`.

## Outputs

One line per arm on stdout with `pp*_tps` / `tg*_tps`. Not evidence authority:
these are single-sample cells for direction-finding, not contract evaluation.

## GPU / build requirements

Exclusive use of GPUs 0 and 1. Binds port 18400 -- not 8080, which llama-swap
owns in production on this host.

## Canonical-state mutation

None. Reads pre-built binaries and an existing tune cache; writes no evidence,
no registry, no cache.

## Safety notes

Each arm launches and kills its own server. `kill -9` is acceptable *here*
because no arm is recording tune measurements -- a tune run must instead shut
down via `/shutdown`, or buffered measurements are silently discarded.

## Disposition

Open. First run (2026-09-06) gave, on a single sample per cell:

| arm | pp256 | tg128 | pp1024 | pp4096 | tg512 | tg2048 |
|---|---|---|---|---|---|---|
| llama-native | 684.36 | 101.63 | 994.99 | 1237.31 | 99.57 | 106.57 |
| bc-control (no winners) | 701.48 | 99.93 | 985.38 | 1238.18 | 100.02 | 106.92 |
| bc-REPLAY (19 winners) | 652.78 | 99.39 | 982.76 | 1220.51 | 98.36 | 105.79 |

Replay is worse than control on all six metrics.

**This result is confounded and must not be cited.** The three arms ran once
each, sequentially, in a fixed order, with replay last. "Replay lost on all
six metrics" is therefore indistinguishable from "the arm that runs third
loses on all six metrics" -- which is exactly what thermal drift over a
~10 minute run would produce. Position and arm are perfectly correlated, so
no amount of squinting at the numbers separates them. The 6/6 direction that
made it look non-random is precisely the pattern the confound predicts.

`replay-bench-balanced.sh` supersedes it: each round runs the three arms in a
rotated order, so over any multiple of 3 rounds every arm occupies every
position equally and monotone drift cancels. Position is recorded per row so
the drift can be tested rather than assumed away.

### Balanced run, 6 rounds / 18 cells, 2026-09-06 (`analyse.py`)

| metric | control | native | replay | native vs ctl | replay vs ctl | ranges |
|---|---|---|---|---|---|---|
| tg128 | 102.08 | 102.11 | 99.97 | +0.02% | **-2.06%** | separated |
| tg512 | 100.25 | 100.26 | 99.04 | +0.01% | **-1.21%** | separated |
| tg2048 | 106.96 | 106.98 | 106.30 | +0.02% | **-0.61%** | separated |
| pp1024 | 985.70 | 987.62 | 980.36 | +0.19% | **-0.54%** | separated |
| pp4096 | 1238.86 | 1237.59 | 1231.41 | -0.10% | **-0.60%** | separated |
| pp256 | 699.66 | 698.00 | 686.33 | -0.24% | -1.91% | overlap |

Two findings.

**The dispatch framework itself is free.** control vs native is +0.02%, +0.01%,
+0.02%, +0.19%, -0.10% across the five clean metrics -- indistinguishable from
zero, which is the target.

**The replay regression is real.** Five metrics show *complete separation*:
every replay sample is worse than every control sample. With n=6 per arm that
is a Mann-Whitney U result at p = 2/C(12,6) ~= 0.0022 per metric. The
confounded first run reached this conclusion by luck; it now stands on a
design that can support it.

pp256 overlaps and carries a 624.57 outlier against a 697.60 median -- it is
startup-sensitive, and per dev-gpt-agent's review must not drive diagnosis.
Position means span only 0.35% (tg128) and 0.19% (pp4096), so drift was not
large enough to have produced the original result on its own -- but it is the
same order as the effect being measured, which is why balancing stays
mandatory.

### Still confounded: winners vs build variant

`replay` and `control` are different binaries -- replay is built with
`GGML_HIP_DISPATCH_REPLAY=ON`. So "the 19 winners are harmful" and "the replay
build variant is slower" both fit the table above. `isolate-winners.sh` runs
the SAME replay binary with the winner cache on and off, holding build variant
constant, and also records MTP draft acceptance so a work-rate difference
cannot masquerade as a throughput difference. Launched 2026-09-06T03:07,
8 rounds / 16 cells.
