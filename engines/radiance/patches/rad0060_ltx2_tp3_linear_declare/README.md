# rad0060: diagnostic LTX-2.5 IQ8R TP3 linear declarations

## Summary

**Unselected diagnostic patch only.** Adds an optional `C01Slice` to
the real Radiance 2.2 / ABI44 `ltx2::Lin::declare` path. Original TP1
call sites have a null slice and remain unchanged. The engine still
rejects `world_size > 1`; this is not an executable TP3 model.

## Shard declarations

- **Fused row banks:** each rank gets the same offset and whole-head width
  from each of the two KV or three QKV banks. Native `LinearFP8::span_parts`
  and `rad_weight_shard_span_parts` apply to code and two-level scale
  planes and to the matching BF16 row bias.
- **Biasless output columns:** `C01Slice::columns` declares a
  `RAD_SHARD_COL` IQ8R K view only at complete, model-derived
  scale-group boundaries, with no bias weight or `table_add`.
- **Biased output columns:** `C01Slice::columns_deferred` additionally
  requires exactly one checkpoint bias source. The partial GEMM
  declares **no bias weight and no bias-add operation**; the native
  `Lin::deferred_bias` state marks its result as *not final*.
  A future rank collective must produce the full-width output and a
  **single owner** must apply that bias once, after the reduction.
  Neither that collective nor its owner is implemented here.

A biased column without explicit deferral, deferral without a bias
source, partial IQ8R scale-group cuts, BF16/small-gate paths and
incompatible fused-bank geometry fail closed before model-weight
declarations. The opt-in patch is never selected by production recipes.

## Evidence and next work

The patched architecture translation unit compiles on the released
2.2 / ABI44 SDK; the host-only, model-backed C01/C02 CTests use actual
`RadFile`, `RadBuilder`, `Lin::declare` and reference kernel schemas.

- 3,024 fused-row/biasless column native declarations pass.
- 2,640 biased output-column native declarations safely defer bias;
  384 model-derived rank K cuts are not admissible.
- Four focused negative cases and every model block's output geometry
  are checked. Source patch applies idempotently (seven edits).

See `docs/evidence/2026-11-22-ltx25-c03-deferred-bias/README.md`.
Native kernel execution, full TP3 DiT graph, all-head Q/K RMS,
collectives, model weight payload loading and video performance remain
unimplemented/unproven.
