# LTX-2.5 TP3 C06 — native source-backed self-attention graph

**Date:** 2026-11-22. **Branch:** `agent/ltx25-tp3-main22`.
**Base:** Radiance 2.2.0 / ABI44, opt-in unselected diagnostic patch `rad0060`.
**Status:** CPU native declaration and host registry proof, **not production or GPU execution**.

## What was reused

This is an optional `C01Slice` and scratch argument to the **existing**
`ltx2::attn_declare` and an optional branch inside the **existing**
`ltx2::attn_core`. The regular TP1 calls pass neither argument and
retain their original operator choices, RoPE parameters and execution
path. The actual LTX source translation unit is patched and compiled
against Radiance's native ABI44 headers; there is no shadow attention
implementation.

For each rank's local self-attention, the diagnostic declares:

1. The original IQ8R `Lin::declare` fused Q|K|V projection with real
   `RAD_SHARD_ROW` and three model-bank spans plus matching BF16 bias spans.
2. The existing full-width BF16 head-gate `Lin` on every rank. Its
   32-head checkpoint weight is small and replicated rather than
   introducing a special gate-shard loader. At `head_gate` issue time,
   the rank uses only its local head-logit columns.
3. Existing C05 full-width Q/K normalization on one shared scratch:
   zero, place local columns, BigCherry exact TP3 `all_reduce`,
   unchanged Radiance `rmsnorm` with full source gain, extract local columns.
4. The original `rope_split` with its existing `head0` parameter
   set to the local rank's first global head. Its `dim` remains the
   **full** video/audio attention width.
5. The original `attn_dense` and `head_gate`, with local head count,
   followed by existing C04 IQ8R output-column projection, exact
   `all_reduce`, then the full BF16 output bias **once per reduced replica**.
6. No additional issue wrapper exists. The original DiT self-attention
   loop already calls input Act::step quantization, Lin::step QKV/gate,
   the same attn_core, output Act::step quantization, and Lin::step
   output reduction in that order. The C06 option changes the core's
   Q/K normalization and makes the existing STG-bypass head-gate
   operand select the rank's local head-logit columns. Missing
   IQ8R quantization is rejected by Attn admission before mutation.

There is **no new transport, rendezvous, kernel, scheduler, GPU selection,
serving authority or configuration format**. The patch remains unselected
by production. The original LTX multi-rank admission refusal is intact.

## Actual source-backed CPU proof

The C06 mode reuses the C02/C04/C05 native test executable, real
`ltx2.cpp`, native `RadFile`/Builder and the ABI-matched isolated
`libref` registry. It reads the real model's fused QKV, gate, Q/K
norm, attention-output weights and biases.

Every one of the **48 DiT blocks** is checked for **video and audio**
self-attention across **8/8/16** and **12/12/8** uneven head partitions
and three ranks: **576 successful complete self-attention builder graphs**.
The test checks:
- IQ8R bank code/scale/bias source spans; output-column scale-group span;
  full unsharded checkpoint norm/gate/output-bias weights
- Real native input and post-attention activation quantizer operations
- Local 32-head ownership coverage, first-head origin and correct
  128-wide video or 64-wide audio head dimensions
- Native `rope_split` parameters `head0`, `n_head` and full `dim`
- Existing exact three-rank `all_reduce` parameters for Q, K and
  output; all declared ops resolve to native host kernels
- Native Builder graph finishing without errors
- Wrong-world, lossy-wire and missing input/output activation
  quantizer requests refused before weight allocation
- Same Q/K scratch handle with no second communication implementation

Source patch applicability: **22 anchored edits**, idempotent on frozen
2.2. The exact native LTX translation unit compiles with GCC14/ABI44.
Native CTests: **5/5** C01, C02, C04, C05 and C06.
The fixture tests retain C01–C05 synthetic applicability and exercise
the complete C06 patch against real source when supplied, rather than
copying fake attention code into the fixture.

## Unproven / next gate

The test declares and resolves native host operations, but **does not
issue them with real activation data**, load the checkpoint payloads
into GPU memory, execute the three-rank collective, compare actual
BF16/quantized outputs against TP1, or generate video. This is the
complete **normal self-attention graph**, not the complete DiT:
cross-attention, cross-modal streams, FFN reductions, modulation,
Full STG guidance correctness and isolated guidance, cancellation
and request epochs remain unintegrated. The original DiT loop contains
the opt-in head-slice handling, but no production caller supplies C06's
TP3 slice, and production world-size admission still rejects TP3.

The C05 zero-packed full-width Q/K all-reduce is correctness-first and
potentially bandwidth-heavy. Its production viability needs an actual
mixed gfx1100/gfx1201 leased-GPU numerical and timing proof. If the
message costs justify compact per-row-statistic normalization, retain
the same Radiance `all_reduce`/serving pipeline and replace only the
within-architecture normalization implementation after measurements.

**Next:** run one real rank-local self-attention component with model
payloads and BF16 TP1 parity, then qualify native three-rank collective
ordering and failure recovery before expanding to all six attention
kinds and the DiT serving owner.
