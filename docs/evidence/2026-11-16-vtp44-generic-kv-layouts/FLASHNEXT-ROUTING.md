# Flash-Next KLD floor: routing changes under any perturbation (VTP44)

## Summary
Flash-Next sends about 1.1% of its routed-expert selections to different experts whenever anything in the arithmetic
changes: a chunk-size change, or any KV cache type. The amount does not depend on the cause, so every arm differs from
every other arm by the same ~22,000 selections, packed caches and the fp8 cache included. This matches the KLD floor
(0.037 to 0.040 for all of them) and shows why KLD against a bf16 reference cannot rank KV caches on this model. It
does not yet prove the routing changes cause the KLD; that needs the per-token routing of each step, which the engine
does not expose.

## Outcome
`qwen3.8-next-flash-hcbf16`, two 7900 XTX TP2 exact wire, no helper, deterministic, one sequence, no speculation,
build `kv-bf16-packed`. The first two documents of the short KLD corpus (4,163 tokens, 48 layers, top-10 routing), prefill
only. Each run counts the routed experts every layer selects (`--expert-stats`) while scoring the same bf16 reference.

| run | KLD mean | p99.9 | top-1 | selections moved vs bf16-a (at least) |
|---|---|---|---|---|
| bf16-a (reference config) | 0 | 0 | 100% | - |
| bf16-b (same config again) | 0 | 0 | 100% | 0 |
| bf16, chunk 128 instead of 64 | 0.0372 | 1.36 | 90.8% | 22,508 (1.127%) |
| bf16_i8 | 0.0382 | 1.58 | 90.8% | 22,052 (1.104%) |
| bf16_fp8 | 0.0380 | 1.56 | 91.4% | 22,052 (1.104%) |
| fp8 | 0.0402 | 1.31 | 90.3% | 22,102 (1.107%) |

Of 1,996,568 selections. "Moved" is half the sum over layers of the L1 distance between two runs' per-layer expert
counts, a lower bound on selections that changed expert (flips that cancel inside a layer are not seen).

- The control pair is exact (0 moved, KLD 0), so the tally is repeatable and a nonzero difference is real.
- The runs differ from each other as much as from bf16-a: bf16_i8 against bf16_fp8 is 22,003, against fp8 22,263 and
  against the chunk-128 run 22,070. The total for bf16_i8 and bf16_fp8 against bf16-a is 22,052 for both by
  coincidence; their per-layer counts differ (layer 3: 118 and 169).
- For the KV arms the first layer that differs is layer 3 and layers 0 to 2 are identical; the chunk-size run differs
  from layer 0. That fits the cache entering at the first attention layer (the model has attention every fourth layer;
  not checked against the layer table here). Moved selections grow with depth, from about 120 at layer 3 to about
  800 at layers 39 to 43.
- At least 459 selections a layer move on average, against 4,163 tokens, so at least 11% of tokens change an expert in
  a typical layer.

## Detail
- This is the arithmetic-noise band of RAD39 seen from the router: any change re-draws about 1.1% of the choices, the
  KLD that follows is 0.037 to 0.040 whether the change is a chunk size or a cache type, and the cache types are not
  ordered by it (bf16_i8 0.0382, bf16_fp8 0.0380, fp8 0.0402 are inside one band).
- Completeness: every run observed all 4,163 tokens in all 48 layers with no dropped histograms. In four layers (34,
  44, 46, 47) the selections counted differ from tokens x top-k, identically in every run including the control, so
  it does not affect the comparison; its cause was not looked at.
- Limits: one document pair, one run per arm, prefill only; tallies, not per-token ids, so causation from routing to
  KLD is inferred. The 27B model has no routing and is not affected.
- What would make KLD usable on Flash-Next: record each step's chosen expert ids in the reference run and replay them in
  the measured runs, so only the cache differs. That needs a per-token expert-id record and a replay option in the
  engine (two manifest fields); it is the follow-up this evidence argues for. It would also give the per-token link
  between a changed expert and the KLD at that token.

Run files (tallies, KLD results, receipts) are in the task's scratch folder `vtp44-fn/routing-keep`; the scripts are
`run_routing.sh` and `routing_diff.py` there.
