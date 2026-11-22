# LTX02: first issued C05 graph with the native Radiance host backend

**2026-11-22.** This is an independently issued three-rank
`RadCtx::run_step` test of the exact **existing** LTX-2.5
`Tp3GlobalRms::declare/step` C05 implementation on Radiance
2.2/ABI44. This original host CTest does not start the GPU.
The same source now also builds an opt-in AQL executable and
passes the model-backed C05 graph on mixed GPUs. There is no
second collective/normalization implementation, no production
TP3 enable, and no complete C06 self-attention block yet.

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

**Follow-up (2026-11-22):** the same native C05 graph source
also passes 1/4/16/64-row host and real Radiance AQL runs
with **2,088,960 / 2,088,960 exact BF16** per backend, using
the original device order and reverse ROCm visibility order.
See [the GPU C05 numerical receipts](../2026-11-22-ltx25-ltx02-issued-aql-rms/README.md).
The original eight-case 122,880-value host table above is the
historical first-pass evidence, not the expanded total.

## Next acceptance gates (still open)

The C05 seeded-activation numerical issue gate now passes on
the actual mixed three-card native AQL engine. The remaining gates
are **captured TP1 checkpoint-activation parity**, actual complete
C06 self-attention (IQ8R QKV, RoPE, gate, attention and post-reduce
BF16 bias), mixed-rank cancellation/refusal and release admission.
Only then can production TP3 be considered. The C05 and C06
production paths have **not** been altered for the tests.

RUM13 owns only the generic native AQL proof; RUM06 owns
shared TP-group lifecycle, and LTX02 alone owns this
model-aware graph and its reference.
