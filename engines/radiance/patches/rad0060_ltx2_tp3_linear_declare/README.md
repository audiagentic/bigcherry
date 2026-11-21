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
  C04 now declares the existing exact Radiance three-rank all_reduce
  followed by a BF16 table_add **once per reduced output replica**.
  No rank adds bias to a partial GEMM. This native host graph does not
  launch a collective or enable production TP3 LTX.

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
  are checked. Source patch applies idempotently (22 edits).

See `docs/evidence/2026-11-22-ltx25-c03-deferred-bias/README.md`.
C04 adds 2,016 source-backed exact reduction and post-reduction bias
declarations. See docs/evidence/2026-11-22-ltx25-c04-native-reduce/README.md.
C05 adds 1,728 native model-backed global-Q/K-norm graph cases
(3,456 Q/K gains) with zero-fill, local cast, exact all_reduce,
the unchanged full-width rmsnorm and local extract. Q and K reuse one
caller-owned scratch buffer; non-TP3, lossy and bad-head settings
refuse. This is a correctness-first, high-bandwidth diagnostic, not
a production kernel or an alternate collective implementation.
See docs/evidence/2026-11-22-ltx25-c05-native-global-rms/README.md.
Actual rank collective execution, full TP3 DiT attention wiring,
model payload loading and video performance remain unproven.

## C06: native self-attention integration

C06 extends the existing Attn::declare / attn_core, optionally.
The new TP3 branch reuses C01/C04/C05, head0 rope_split, attn_dense,
head_gate and the original Act::step quantizers. The 32-head BF16 gate
projection is replicated, avoiding a new small-weight shard loader.
The same attn_core issues attention for both TP1 and diagnostic TP3,
preserving the original single-rank operations.

The model-backed ABI44 test checks 576 video/audio self-attention
builder graphs (48 blocks x 2 modalities x 2 legal splits x 3 ranks).
The existing DiT step loop is reused (including STG gate slicing), but no owner selects TP3: no GPU execution, model payloads,
full DiT or TP3 serving enabled. See
docs/evidence/2026-11-22-ltx25-c06-native-self-attn/README.md.
