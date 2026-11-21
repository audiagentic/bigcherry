# LTX-2.5 C04: source-backed exact TP3 output reduction and BF16 bias finalization

**Date:** 2026-11-22. **Branch:** `agent/ltx25-tp3-main22`.
**Baseline:** released Radiance 2.2.0 / ABI44; diagnostic-only `rad0060_ltx2_tp3_linear_declare`.

## Reuse of existing BigCherry TP3

BigCherry main already implements mixed gfx1100/gfx1201 TP3 through per-rank
code selection (`rad0016`), shared collective rendezvous (`rad0017`),
uneven ownership (`rad0018/0019`), and exact three-rank collective
improvements (`rad0021`, `rad0022`, `rad0024`, `rad0032/0033`).
C04 **does not implement a new collective transport**.

The opt-in LTX diagnostic patch now exposes
`ltx2::Lin::finalize_deferred_output` for a successfully declared
IQ8R `columns_deferred` projection. It declares the existing Radiance 2.2
`all_reduce` operation with `world_size=3` and `exact=1`, then declares
the original `table_add` BF16 bias operation. At step issue time, the
operations are ordered local partial GEMM → exact all-reduce → bias.
Each rank receives a full output replica and adds the checkpoint bias
**once to that replica**; no rank adds it to its partial GEMM.
This is per-replica finalization, not a separate rank-0 broadcast path.

The finalizer rejects non-deferred linear projections, non-TP3 worlds,
duplicate finalization, invalid shape and missing prior projection;
the C03 preflight continues to reject biased column shards without a
single checkpoint bias source or without 256-aligned IQ8R K spans.
The LTX `world_size > 1` production admission refusal and TP1 code
remain intact. No production source recipe selects this diagnostic patch.

## Measured CPU-only proof

The patched real 2.2 LTX source translation unit compiles on the native
ABI44 headers. The model-backed C04 test uses the real, read-only IQ8R
RAD2 model, actual `ltx2::Lin::declare`, and a native `RadBuilder`.
It checks all model DiT blocks and the two scale-group-valid uneven
head layouts (8/8/16 and 12/12/8) on all three ranks. The seven biased
output projection types are video/audio self-attention, both text
cross-attention outputs, both cross-modal outputs and audio FFN-down.

C04 checks, for each source-backed declaration: no rank-local bias
before reduction; full-width, unsharded BF16 final bias; exact all-reduce
op and native reference-kernel resolution; builder finalization; and
duplicate finalizer refusal. A non-TP3 world is rejected without mutating
the builder. The 10/10/12 audio K cut is retained as a negative test.
C02 independently covers all three proposed head layouts and the full
48-block invalid-K census (384 rank cuts). There is no presumption that
a 10/10/12 audio projection can be made valid by changing the collective.

**Proof boundary:** C04 builds a real source-backed, native **host op
graph** with the existing all-reduce API; it does not launch a
three-rank collective, load model payloads onto GPUs, execute attention,
produce video or show numerical TP1 parity. Both the complete TP3 LTX
activation graph and the global all-head Q/K RMS operator remain undone.
The three-rank kernel backend, mixed-device lifecycle and cancellation
still require hardware qualification for LTX's graph and message sizes.

## Reproduction

Stage an *empty*, isolated source directory with
`tools/tests/lab/test_ltx_tp3_builder_patch.py` against the pinned
2.2 source. Configure `tools/lab/radiance/ltx-tp3/native-c01-gate`
with `LTX_SOURCE_ROOT`, `LTX_STAGE_ROOT`, `LTX_NATIVE_CORE_LIB`,
`LTX_REFERENCE_LIB` and `LTX_IQ8R_MODEL` matching the release.
Build and run the `test` target, which includes the new
`ltx_c04_deferred_output_reduce` test. The separate synthetic 11-CTest
LTX suite remains independent of the actual model and kernels.

## Next implementation

Connect the proven C04 finalizer to a real rank-local attention-block
builder, with global Q/K RMS, per-rank buffers and matched collective
schedule. Only then do a short leased mixed-GPU numerical comparison
against TP1. Production LTX TP3 admission stays disabled until the
whole graph, model load, guidance and cancellation pass.


## Validation recorded in this slice

- Nine anchored source edits apply idempotently against pinned 2.2.
- Patched native ABI44 LTX translation unit compiles with GCC 14.
- Three model-backed native CTests: 3/3 pass (C01, C02 and C04).
- C04 host graph declarations: 2,016 positive, invalid audio K refused,
  wrong-world refusal without builder mutation, duplicate finalization refused.
- C02: 3,024 fused/biasless positive, 2,640 deferred positive,
  384 invalid K cuts refused; original four negative probes retained.
- Eleven synthetic LTX native CTests: 11/11 pass.
- Focused Python lab fixtures: 4 pass, 2 optional model/source skips.
- Core tooling boundaries, engine literal and model constant guards:
  23 pass. Git diff whitespace check passes.

These are source-backed and/or CPU test results, **not** GPU validation.
Planning and Ledger connector actions were unavailable to this session;
no authoritative planning or release events were synthesized.
