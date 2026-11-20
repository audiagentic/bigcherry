# rad0060: LTX-2.5 IQ8R C01 native linear declaration (diagnostic only)

## Summary

An opt-in, diagnostic-only C01Slice for the LTX2 `Lin::declare` path (details below); TP1 behaviour is unchanged and the
`world_size > 1` refusal is preserved.

## Outcome

See SUMMARY.md: compile and applicability evidence only; nothing here is selected by a production patch set.

## Detail

**Status:** native staging slice, source-pinned to Radiance's frozen 2.2 / ABI44
candidate. Not selected by any production patch set or engine launch. This
package creates `arch/ltx2/bc_ltx_tp3_c01.h` and changes only
`arch/ltx2/ltx2.cpp::Lin::declare` to accept an *optional* `C01Slice`.
Every existing caller passes no slice and keeps exactly the original TP1
behavior. The existing rejection of `world_size > 1` is untouched.

C01's Rows form records whole-head [start, start + width) in **each** of
two KV or three QKV source banks. It declares each native IQ8R weight and
its `scale,scale.1` logical view as a `RAD_SHARD_ROW` weight with per-bank
parts; `LinearFP8::span_parts` and `rad_weight_shard_span_parts` record
the same model-source bank offsets for codes/scales and the BF16 bias.
A plain contiguous slice of fused QKV would be incorrect.

C01's Columns form declares an aligned IQ8R `RAD_SHARD_COL` weight
using `LinearFP8::declare` with [start, start + width); **only biasless
outputs** are admitted. Biased attention outputs need an explicit
once-after-collective bias owner and are refused at this stage. Invalid
geometry, a 128-wide/partial scale group, plain/other encoding or a
small-bf16 gate are refused.

No rank-local activation buffer, all-reduce, Q/K RMS, graph schedule or
serving/owner path is introduced by this patch. An actual production
activation needs all those dependencies, plus identity and GPU numerical
evidence. This patch's inclusion in a diagnostic source, separately from
the production recipe, must be explicitly reviewed.

## Proof gates

The lab's `ltx_tp3_builder_shard_test` compiles the **same** header from
this package and tests head/scale/bank admission. The actual Radiance
`ltx2.cpp` patch requires source-applicability and host syntax/ABI
compile checks; passing the header test alone is not a builder proof.
See `docs/design/ltx25-tp3-component-test-ladder.md`.
