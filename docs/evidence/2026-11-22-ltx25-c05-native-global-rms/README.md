# LTX-2.5 C05: existing-op global Q/K RMS, native host graph

**Date:** 2026-11-22. **Branch:** `agent/ltx25-tp3-main22`.
**Baseline:** released Radiance 2.2.0 / ABI44.
**Patch:** unselected `rad0060_ltx2_tp3_linear_declare`.

## Smallest reuse-first design

LTX 2.5's `attn_declare` and `attn_core` use `rmsnorm` over the
**full 4096 video / 2048 audio channels** before RoPE. Each model Q/K
gain has that same full width. Per-rank RMS over 8, 12 or 16 owned
heads is **not** equivalent. The current upstream registry has no
primitive that accepts a globally reduced square sum for an RMS apply.

The diagnostic `ltx2::Tp3GlobalRms` is inserted into the **real**
architecture translation unit, but has **no production callers**.
For each Q or K it issues native operations in this order:

1. `fill`: zero the caller-owned full-width BF16 scratch.
2. `cast`: copy the rank's compact owned-head columns into their
   global offsets; other columns remain zero.
3. `all_reduce`: existing exact three-rank collective. Non-overlapping
   columns reconstruct the full Q/K row on every rank.
4. `rmsnorm`: the existing unmodified TP1 operator with the
   checkpoint's full-width replicated BF16 gain and epsilon.
5. `cast`: extract the normalized local columns for rank-local
   RoPE and attention.

The same scratch allocation is reused by Q and K **sequentially**;
the helper does not create another scratch owner, rendezvous,
collective, config parser or kernel. The `Geom` is the authority for
world and wire configuration. It rejects non-TP3 worlds, lossy wire,
fractional-head spans, invalid sizes/epsilon and repeat declaration
without allocating a duplicate operation. The step fails through the
existing `rad_step_fail` if required operators were not declared.
The `C01Slice` and native `Lin::declare` remain responsible for
IQ8R fused-bank row projection and aligned K/scale views.

## Source-backed host proof

The C05 mode of the existing C02/C04 test includes the freshly patched
real LTX source and uses the ABI-matched native `Builder`, `RadFile`
and the isolated `libref` registry. It checks **all 48 DiT blocks**,
the 6 fused QKV/KV attention projection kinds, two IQ8R-admissible
whole-head splits (8/8/16 and 12/12/8), and three ranks:

- **1,728** fused-projection graph cases.
- **3,456** Q/K full-width RMS declarations and model gain validations.
- **17,280** existing Radiance op declarations
  (two five-op normalizers per case).
- Every declared op resolves to a native host reference kernel and the
  Builder graph finalizes. Q and K share one scratch handle.
- Non-TP3, lossy-wire, fractional-head and duplicate-declaration
  requests refuse. No unsupported audio 10/10/12 K projection is
  admitted.

The 2-bank KV cases use a synthetic local-Q activation buffer while
checking the actual model's Q and K gains and real KV projection.
These cases **do not** prove an integrated cross-attention Q projection.
The self-attention cases include real native fused-QKV declarations.
The separate existing C++ RMS test now compares the **zero-pack,
rank-reduce, normalize, extract** arithmetic against both full-width
dense RMS and the independent compact-square-sum TP3 oracle for
video/audio widths, Q/K row counts and three uneven splits.

## Performance and correctness limits

This simple composition is appropriate as a **baseline for
correctness**, not as a finalized performance plan. It all-reduces
full BF16 Q and K rows: at 4096-wide video each Q/K pair carries
2 × 4096 × 2 = **16 KiB per row per rank** of collective input,
and at 2048-wide audio, **8 KiB**. These are message payload sizes,
**not** network transfer estimates. Packing, clearing, both copies,
kernel launches and extra full-width scratch also cost time.
A compact per-row square-sum all-reduce plus native RMS-apply kernel
might later replace this implementation **inside the same architecture
helper**, only if measured throughput warrants the new kernel.
Do not build a separate LTX collective or serving path.

**No real kernel issues, model payload loading, GPU numerical parity,
full `Attn::declare` wiring, RoPE, attention, output generation,
guidance or cancellation has yet been executed here.** The existing
LTX world-size refusal remains intact, and no production recipe
selects the diagnostic patch.

## Verification

- Ten source patch edits apply idempotently against frozen Radiance 2.2.
- The patched LTX translation unit compiles with ABI44/GCC 14.
- Native source-backed CTests: **4/4** (C01, C02, C04, C05).
- Synthetic native tests: **11/11**, including extended packed-RMS oracle.
- Fixture Python patch tests: **2 passed**, **1 optional skip**.
- Additional engine/tooling guards are tracked by the parent review.

## Next gate

Integrate **one real self-attention projection → normalized Q/K →
local RoPE/attention/gate → C04 reduced output** into a single
source-backed owner graph, with identical collective issue order on
every rank. Measure BF16 transfer and kernel capability on the
existing mixed-GPU TP3 stack under a lease before treating this
full-width baseline as production-viable. Reuse native scheduler,
configuration, rank mapping and resource admission unchanged.
