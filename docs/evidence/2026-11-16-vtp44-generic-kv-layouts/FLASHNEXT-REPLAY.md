# Flash-Next KV cache KLD with the routing replayed (VTP44)

## Summary
Replaying a reference run's expert choices (rad0044 records them, rad0045 replays them) removes the routing noise that made
KLD useless for ranking KV caches on Flash-Next. With the route fixed, the cache types separate: bf16_i8 is
indistinguishable from the arithmetic noise of a chunk-size change (0.0074 against 0.0076), bf16_fp8 is slightly above it
(+0.0008) and fp8 is clearly above it (+0.0021). Without replay all of them sat at 0.037 to 0.040. One document pair, one
run per arm.

## Outcome
`qwen3.8-next-flash-hcbf16`, two 7900 XTX TP2 exact wire, no helper, deterministic, one sequence, no speculation, build with
rad0043/0044/0045 (experiment `expert-route-replay`). The first two documents of the short KLD corpus (4,163 tokens, 4,161
scored positions) against the same bf16 reference logits as FLASHNEXT-ROUTING.md. Every run replays the route recorded by the
reference-configuration run (bf16, chunk 64).

| run | KLD mean, replayed | KLD mean, not replayed | top-1, replayed | top-1, not replayed |
|---|---|---|---|---|
| bf16, chunk 64 (the reference's own configuration) | 0 | 0 | 100% | 100% |
| bf16, chunk 128 instead of 64 | 0.00762 | 0.0372 | 95.7% | 90.8% |
| bf16_i8 | 0.00738 | 0.0382 | 95.6% | 90.8% |
| bf16_fp8 | 0.00844 | 0.0380 | 95.2% | 91.4% |
| fp8 | 0.00971 | 0.0402 | 94.9% | 90.3% |

- Replay works: in every replayed run the per-layer expert counts equal the record's histogram in all 48 layers, and
  replaying the reference's own record gives KLD exactly 0.
- The routing noise was about 80% of the KLD: the chunk-size change alone falls from 0.0372 to 0.0076. What remains is the
  arithmetic noise of the same experts, and that is the reference the caches are read against.
- Paired bootstrap over positions (2,000 resamples, 95%; `flashnext-replay/bootstrap-output.txt`):

  | difference | mean | 95% interval | |
  |---|---|---|---|
  | bf16_i8 - chunk-128 reference | -0.00024 | -0.00094 to +0.00037 | includes 0 |
  | bf16_fp8 - chunk-128 reference | +0.00082 | +0.00011 to +0.00143 | excludes 0 |
  | fp8 - chunk-128 reference | +0.00209 | +0.00120 to +0.00289 | excludes 0 |
  | bf16_fp8 - bf16_i8 | +0.00106 | +0.00057 to +0.00158 | excludes 0 |
  | fp8 - bf16_i8 | +0.00232 | +0.00171 to +0.00300 | excludes 0 |
  | fp8 - bf16_fp8 | +0.00126 | +0.00060 to +0.00193 | excludes 0 |

## Detail
- What replay forces and what it does not: the experts each token's router chose, in every routed layer. The routing weights
  are the measured run's own softmax over its own logits at those experts, and everything else (the arithmetic, the cache) is
  the measured run's. That was enough for an exact self-replay, so recorded weights are not needed.
- What it hides: a cache that changes the router's choices in production would change them differently from the reference.
  The replay isolates the cache's effect on the logits for the same experts, which is the quantity the earlier KLD criterion
  meant to measure; the routing flips themselves (about 1.1% of selections under any perturbation, FLASHNEXT-ROUTING.md) are
  the same size for every arm and are not an argument between caches.
- Reading the numbers: the caches sit 0.007 to 0.010 above zero, of which about 0.0076 is the arithmetic noise a
  chunk-size change makes on its own. bf16_i8 adds nothing measurable to it; bf16_fp8 and fp8 add 0.0008 and 0.0021.
- Limits: one document pair of 4,163 tokens, one run per arm, prefill only, TP2 on two gfx1100 cards (the R9700 engine
  layouts were not scored this way). Positions are not independent, so the intervals are optimistic; a second corpus slice
  would show how much of the ordering is the documents'. The 27B dense model has no routing and is not affected.
- What it does not show: decode-time effects, long-context behaviour, or the bf16_i8 one-stream decode difference noted in
  FLASHNEXT-HELPER-AB.md.

## Files
`flashnext-replay/`: the five replayed runs' KLD files and the four not-replayed ones they are compared with
(`noreplay-*`), receipts, `run_record.sh` / `run_replay.sh` (the runs; paths are those of the run), `verify_record.py` /
`verify_replay.py` and their outputs (record against stats counts; replay against the record and the not-replayed KLD), and
`bootstrap_replay.py` with its output. The per-position rows, the stats files and the 8 MB route file were in the run
folders and are not kept; the route file is reproduced by `run_record.sh`.
