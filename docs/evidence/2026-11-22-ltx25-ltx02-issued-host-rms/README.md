# LTX02: first issued C05 graph with the native Radiance host backend

**2026-11-22.** This is an independently issued three-rank
`RadCtx::run_step` test of the exact **existing** LTX-2.5
`Tp3GlobalRms::declare/step` C05 implementation on Radiance
2.2/ABI44. It does **not** create another collective/normalization
implementation, start the GPU, enable LTX TP3 serving, or establish
a complete C06 DiT self-attention block.

## Source and binding

- Reads the actual, read-only LTX-2.5 IQ8R RAD2 container through
  upstream `rad::RadFile`; model-derived QKV width, head dimension
  and **real BF16 Q/K gain planes** from first video/audio DiT block.
- Declares the existing C05 fill, local insert, exact native
  `all_reduce`, full-width `rmsnorm` with each real gain, and
  local extract. Executes through native `RadBuilder`, `Program`,
  `rad::Ctx` and upstream `libref` registry, not a handwritten
  operation simulator.
- Three distinct native **rank contexts** (all on the host backend)
  issue two collective operations each. All-width normalization
  remains 4096 video / 2048 audio, source-derived and independent
  of the uneven local Q/K head ownership.
- Deterministic BF16 **synthetic test activations** were chosen to
  isolate the native execution/collective. They are **not**
  checkpoint intermediate activations. The independent host oracle
  reconstructs the full width in ascending-column f32 accumulation,
  then applies the checkpoint gain with one BF16 rounding.

## Numeric receipt

| Native graph | Legal heads | Rows | Exact BF16 comparisons |
| --- | --- | ---: | ---: |
| Video Q+K | 8/8/16 | 1 | 8192/8192 |
| Video Q+K | 12/12/8 | 1 | 8192/8192 |
| Video Q+K | 8/8/16 | 4 | 32768/32768 |
| Video Q+K | 12/12/8 | 4 | 32768/32768 |
| Audio Q+K | 8/8/16 | 1 | 4096/4096 |
| Audio Q+K | 12/12/8 | 1 | 4096/4096 |
| Audio Q+K | 8/8/16 | 4 | 16384/16384 |
| Audio Q+K | 12/12/8 | 4 | 16384/16384 |
| **Total** | | | **122880/122880** |

The first *failing* graph issue returned 77/8192 exact BF16 in
the first row, with all 3 ranks and both native collectives
issuing. Investigation showed the test's Q and K input buffers
could share the same **liveness-reused transient arena region**:
the test writes both *before* running the declared graph, whereas
transient liveness assumes their producers follow graph order.
Declaring those externally seeded **test inputs**
`RAD_BUF_PERSIST` and explicitly refusing Q/K pointer equality
eliminated the alias. The production `rad0060` C05 code was
not changed to fix a fixture ownership mistake.

## Reproduction

The native test is registered in the existing
`tools/lab/radiance/ltx-tp3/native-c01-gate/CMakeLists.txt`
as `ltx_c05_issued_host_global_qk`; it uses the *same*
staged patched LTX2 translation unit, real model source,
host-enabled Radiance 2.2 core and isolated native libref
already used for C01–C06 declarations. The test has a
bounded CTest timeout. No GPU lease is needed.

## Next acceptance gates (still open)

1. Use the actual HIP-enabled 2.2 `RadCtx` AQL backend,
   rank-code-object selection, real device allocations and
   exact native collectives on **three distinct GPUs**.
   Check checkpoint gains, model-derived Q/K geometry, native
   per-rank output parity, kernel coverage, cancellation
   and issue order under short acquired GPU leases.
2. Replace the synthetic activation oracle with captured
   TP1 checkpoint activation intermediate(s) when available,
   and verify 1/16/64-row native graph parity for both
   modalities and legal layouts.
3. Extend the same test to actual full C06 self-attention
   (IQ8R QKV, RoPE, gate, attention, exact output reduce,
   full-width bias applied once). Production TP3 remains
   refused until complete model parity and serving admission.

RUM13 owns only the generic native AQL proof; RUM06 owns
shared TP-group lifecycle, and LTX02 alone owns this
model-aware graph and its reference.
