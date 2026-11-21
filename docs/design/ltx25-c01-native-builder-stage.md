# LTX-2.5 TP3 — C01 native Radiance builder stage

**Status 2026-11-20:** diagnostic patch implemented, source applicability
and host native translation-unit compile proven; **no TP3 engine
execution or real RadBuilder weight-admission run yet**.

## Ownership

`engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/` introduces
`arch/ltx2/bc_ltx_tp3_c01.h` and an optional `const C01Slice*`
argument to the actual `arch/ltx2/ltx2.cpp::Lin::declare`.
No existing call site passes this argument. The canonical recipe
`patch-set.ltx3-c01-diagnostic` references this unpromoted package,
but **no source recipe selects that set**. Production v1.17.0 and
the LTX `world_size > 1` refusal are unchanged.

## Native behavior introduced

- **Fused row weights:** for a QKV (three banks) or KV (two banks),
  `C01Slice::rows(full_bank, offset, width, head_dim, bank_count)`
  checks the whole-head rank slice. The native `LinearFP8::declare`
  uses `RAD_SHARD_ROW`, its row parts and `span_parts`; the fused
  BF16 bias uses the corresponding real `rad_weight_shard_span_parts`.
  IQ8R's logical `scale,scale.1` view tracks the code-plane spans.
  The source part count and full-bank sizes must match; mismatches
  refuse the declaration, never take a contiguous fused slice.
- **Output-column weights:** `C01Slice::columns(full_K, offset,
  local_K)` uses `RAD_SHARD_COL` and the existing
  `LinearFP8::declare` span to select the local 256-aligned IQ8R K
  slice. **Only biasless output projections are supported in C01.**
  Attention outputs with bias are refused rather than adding bias
  on all ranks. In the actual LTX model, video `ff_down` is
  biasless; the audio FFN-down and six attention outputs are biased.
- **Conservative admission:** unsupported BF16/small-gate paths,
  non-IQ8R groups, mismatched bank/input declarations, invalid
  heads, incomplete scale groups, malformed geometry and
  biased output-column requests are explicitly refused.
- **TP1 unchanged:** `slice == nullptr` retains the original
  unsharded `LinearFP8` branch and bias declaration.

## How to reproduce host gates

Run from a BigCherry worktree and use the approved build root.
The paths below are project-relative / environment variables;
Remote Agent calls can use absolute equivalents.

```sh
pytest -q tools/tests/lab/test_ltx_tp3_builder_patch.py
cmake -S tools/lab/radiance/ltx-tp3 -B "${BC_WORK_ROOT}/ltx-tp3-native-build" -DBUILD_TESTING=ON
cmake --build "${BC_WORK_ROOT}/ltx-tp3-native-build" --target test
```

With a separately staged, frozen 2.2 Radiance source,
`test_ltx_tp3_builder_patch.py` accepts the input source and
an **empty output directory** to copy only the LTX translation unit and
apply the anchored package. It proves idempotence. The
`tools/lab/radiance/ltx-tp3/native-c01-gate` CMake project compiles
the **patched real LTX2 source translation unit** with its frozen
ABI44 headers, without linking or running a model. Configure it with
`-DLTX_SOURCE_ROOT=<source>`, `-DLTX_STAGE_ROOT=<output>`.
The staged path is outside source control and is never a serving
artifact.

`ltx_tp3_builder_shard_test` compiles the actual C01 header shipped
inside the patch package. Alongside the existing 10 CTests, this
checks all three candidate partitions for 2-/3-bank fused rows
and 256-aligned output K cuts, including rejected audio 10/10/12 and
oversized/invalid source spans.

## Not yet proven

- The 2026-11-21 native `RadBuilder` C01 weight-view receipt covers
  actual model IQ8R codes, logical scale views and BF16 row bias.
  It does **not** prove complete graph/kernel resolution, payload load or GPU execution.
  See `docs/evidence/2026-11-21-ltx25-2p2-native-builder/README.md`.
- No real rank-local activation buffer, global all-head Q/K RMS,
  RoPE/attention/FFN output collective or bias-once ownership.
- No matching mixed gfx1100/gfx1201 kernel and hardware numerical
  comparison, no model output, no device liveness or E2E parity.
- No authoritative Planning/Ledger closure or production promotion.

## Next single proof

The native C01 model-weight **declaration** receipt is now covered,
including all DiT blocks and the actual two-level IQ8R scale views.
Next, run the real LTX `Lin::declare` for one source-bound layer with
declared activation buffers, native kernel-selection and full
weight-loading receipts. Compare a single-layer numerical projection
against TP1 before permitting any TP3 serving admission. Biased
attention outputs require an explicitly unique after-reduction owner.
See `docs/evidence/2026-11-21-ltx25-2p2-native-builder/README.md`.
