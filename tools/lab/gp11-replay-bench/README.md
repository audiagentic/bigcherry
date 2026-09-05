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

Replay is worse than control on all six metrics. Needs repeats before it is
called a result; recorded here so the direction is not lost.
