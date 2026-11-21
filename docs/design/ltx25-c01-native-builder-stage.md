# LTX-2.5 TP3 — C01 native Radiance builder stage

**Status 2026-11-22:** diagnostic patch compiled against the released 2.2/ABI44 source; native C01 model-derived weight receipts and C02 actual `Lin::declare` host graph finalization passed. **No full model payload loading, TP3 serving or GPU generation** has been demonstrated.

## Ownership

`engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/` introduces
`arch/ltx2/bc_ltx_tp3_c01.h` and an optional `const C01Slice*`
argument to the actual `arch/ltx2/ltx2.cpp::Lin::declare`.
No existing call site passes this argument. The canonical recipe
`patch-set.ltx3-c01-diagnostic` references this unpromoted package,
but **no source recipe selects that set**. Production Radiance 2.2.0 and
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
  slice. The ordinary `columns` form remains **biasless**.
  The opt-in `columns_deferred` form requires one bias source but
  declares no per-rank bias weight or bias-add; `Lin::deferred_bias`
  requires a later unique post-reduction owner (not implemented).
  In the actual LTX model, video `ff_down` is biasless; the
  audio FFN-down and six attention outputs are biased.
- **Conservative admission:** unsupported BF16/small-gate paths,
  non-IQ8R groups, mismatched bank/input declarations, invalid
  heads, incomplete scale groups, malformed geometry and
  biased columns without explicit safe deferral, and deferred
  columns without a bias source, are explicitly refused.
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
ABI44 headers. With an explicit ABI-matched native core and reference
kernel, C01/C02 additionally test actual model-backed declarations and
host graph finalization; no GPU execution is enabled. Configure it with
`-DLTX_SOURCE_ROOT=<source>`, `-DLTX_STAGE_ROOT=<output>`.
The staged path is outside source control and is never a serving
artifact.

`ltx_tp3_builder_shard_test` compiles the actual C01 header shipped
inside the patch package. Alongside the existing 10 CTests, this
checks all three candidate partitions for 2-/3-bank fused rows
and 256-aligned output K cuts, including rejected audio 10/10/12 and
oversized/invalid source spans.

## Not yet proven

- Native C01 model IQ8R weight views and C02 real `Lin::declare` host
  graphs are proven; they do **not** establish complete TP3 architecture
  graph declaration, model payload loading or GPU execution.
  See `docs/evidence/2026-11-22-ltx25-main22-c02/README.md`.
- No integrated rank-local activation scheduling, global all-head Q/K RMS,
  RoPE/attention/FFN output collective or bias-once ownership.
- No matching mixed gfx1100/gfx1201 kernel and hardware numerical
  comparison, no model output, no device liveness or E2E parity.
- No authoritative Planning/Ledger closure or production promotion.

## Next single proof

**C02/C03 host-only declarations:** 3,024 original fused/biasless `Lin::declare` calls plus 2,640 safely deferred-biased output columns across all 48 DiT blocks, seven output types, three head splits and three ranks; 384 rank K cuts are incompatible with IQ8R group alignment. Four explicit invalid requests fail before model-weight declarations. Native host kernel resolution and graph finalization pass. Evidence: `docs/evidence/2026-11-22-ltx25-c03-deferred-bias/README.md`.

Next, integrate all-head Q/K RMS, rank-local buffers and bias-once
output reductions into one real LTX attention block. Validate its native
model payload load and numerical output against TP1 before permitting
any TP3 serving admission. Full dual-stream DiT and device performance
remain later gates.


## C04 exact output reduction graph (2026-11-22)

The diagnostic-only Lin::finalize_deferred_output combines an existing
Radiance 2.2 exact world-three all_reduce with a BF16 bias-add once on
each full reduced output replica. This reuses BigCherry TP3 collectives,
without a new transport or any production LTX TP3 admission change.

The native C04 test covers all 48 DiT blocks, both aligned uneven
head splits, all three ranks and seven biased output types. It rejects
duplicate finalization, non-TP3 worlds and audio K misalignment.
See docs/evidence/2026-11-22-ltx25-c04-native-reduce/README.md.
GPU execution and TP1 numerical parity are not established.
