---
id: VTP32
order: 32
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-12T07:38:53.993312+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# Per-op-kind split tuning: measure each card's cost per layer kind and solve the tp_ownership split

## Description

The cards do not run every layer kind at the same rate. gfx1100 emulates fp8, so an XTX is much slower than the R9700 on MXFP4/fp8 GEMMs but runs int8 natively. Decode GEMMs at small M are memory-bound (XTX 960 GB/s against R9700 ~640 GB/s), while prefill GEMMs are compute-bound (the R9700 leads on fp8). Attention, the delta-net scan and the feed-forward each have their own cost per card.

Tensor parallelism splits within a layer, never by layer: every rank holds all 64 layers. `--tp-ownership` (rad0018/rad0019; manifest `tuning.tp_ownership`) already sets the share of each layer KIND independently:
- `kv` (and `q` = kv x 6): the 16 attention layers;
- `lk`/`lv` (v = 3k on each rank): the 48 delta-net layers;
- `ffn`: every layer's feed-forward rows.

Instances of one kind have identical shapes, so per-instance shares would not help. The kind is the right granularity, and finer granularity inside a layer is fixed by the column-then-row parallel pairs (gate/up with down, heads with o_proj).

This item turns the hand sweeps into a solver. Measure each card's cost per op kind and per regime, then choose the shares that minimise the sum over layers of the slowest rank's time (compute plus its all-reduce wait). Each all-reduce waits for the slowest rank of that layer, so per-rank totals can balance while per-layer waits still add up.

## Results so far

- MXFP4, `--profile-ops`, one stream depth 7: XTX compute 3468 ms against R9700 2263 ms at 6/6/12. Bandwidth weighting (more on the XTX) made the XTX +10% slower.
- MXFP4 sweep (4/8 streams, spec off and depth 3): 6/6/12 had the shortest step in 3 of 4 regimes.
- w8a8 sweep: delta-net 6,6,4 and FFN 6656,6656,4096 is +13% at 8 streams (spec off) and +7% (spec on). Int8 favours more work on the XTX.
- KV head placement (w8a8): 1,1,2 beats 2,1,1 on capacity (12.15 against 11.72 GiB) and speed.
- The optimum differs by regime (decode against prefill) and by weight format. A single static split must compromise: the split is fixed at load because the weights are resident per rank.

## Steps

1. A calibration run per card, op kind and regime: `--profile-ops` per-op ms and calls (and the wait column) at decode (1/4/8 streams) and prefill (2048/8192 tokens), MXFP4 and w8a8.
2. A cost model per rank: t_kind(rank, share, M) from the calibration, plus the all-reduce cost per call (VTP31).
3. A solver (tools/bigcherry, card-free): integer shares under the constraints (whole KV heads, q = 6 kv, lv = 3 lk, FFN in the weight format's block multiples, sums equal to the model's totals). It minimises the workload-weighted sum over layers of the per-layer maximum, and emits the `tuning.tp_ownership` spec and a layout settings line.
4. Validate the solver's split against the best hand split by ABBA, per format.
5. Per-workload profiles: a latency (decode-weighted) and a throughput (prefill-weighted) split as separate layouts; a run picks one in its manifest.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Card-free unit tests of the solver's constraints (refuses an infeasible model, sums and ratios hold).
- ABBA solver split against the hand split, per format, at 4/6/8 streams with speculation on.

## Effort & Risk



## Standards



## Acceptance Criteria

- The solver emits a valid spec for any model and card set from calibration data alone (no model or card named in code).
- On MXFP4 and w8a8 its split is within noise of, or better than, the best hand split at 4/6/8 streams.
- Splits are chosen per layout or run, never compiled in.

## Notes

Hand-sweep data: VTP14.md (N2 rebalance, split sweep, N6). Profile tooling: ${BC_TASK_TMP}/vtp14/prof_split.py, prof-where.sh. Spec form and checks: rad0018 README.

## Change Log

- 2026-11-12T07:38:53.993312+00:00 (created-by): Created by agent
- 2026-11-12T07:40:13.909697+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
