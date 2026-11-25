# RCS15, first part: what upstream's native RDNA3 library covers at the pinned release

Source inspected: the pinned upstream release (2.2.0, also the newest tag at the time), read from the local mirror of upstream; no build and no GPU. This is the GPU-free part of the baseline; the matched measurements (arms A, B, C) are the second part.

## What the native RDNA3 library is
- A separate upstream library, `libr3d`, for gfx11 only (`^gfx11`), with its own units (51 build units named `r3d_*`) and its own row table. It is not libr4d compiled for gfx11.
- Its kernels rely on wave32 and RDNA3 instructions: dot2 accumulate and the 16x16x16 wave-matrix family; the GEMM rows carry two arms each (dot2 pairs and wave matrix) selected by the tune cache per shape per machine. It decodes every quantised weight in software and uses no gfx12-only instruction.
- libr4d (gfx12) has 115 build units; the two libraries do not share unit names.

## Coverage of the formats we serve
Counting source files that mention each form at the pinned release:

| Form or family | libr3d files | libr4d files |
|---|---:|---:|
| IQR (trellis-coded 6/7-bit weights) | 0 rows (one comment mentions IQR dtypes) | 24 files, including the GEMM units (`r4d_gemm_iqr*`), the decoder header, the embedding gather, the vocabulary head, the MoE variant |
| local int8 form with rotated activations (the IQ8R container) | 0 | in the int8 GEMM units |
| MXFP4 | 13 | present |
| E4M3 | 33 | present |
| GDN | 23 | present |
| MoE | 26 | present |
| DFlash | 11 | present |

## Consequences for the plan
1. For the two formats we actually serve in the quality-first configurations (IQ8R and IQR), upstream's native RDNA3 library has no row. A native-provider cutover would not replace our gfx11 IQR and IQ8R path; it would need those rows added, which is what the owned overlay plugins (RCS23) are. So the overlay model is needed under either outcome of the RCS15 decision.
2. The decision RCS15 still has to make is about the rest (MXFP4, E4M3, GDN, MoE, DFlash, attention, helpers): whether the native library is at least as good as libr3 for those rows. That needs the matched arms, not source reading.
3. Reading the source is not evidence of support: a row's presence says nothing about shape coverage, ragged tails or speed. The matrix still has to be built from selected rows on real shapes.

## Next
Arms A (our gfx11 build of libr4d's units), B (upstream native library, stock) and C (the reference oracle) on one XTX with the first-part inventory as the row list; the IQR and IQ8R rows are marked "no native row" and excluded from B.

## Update: upstream main, 119 commits past the pinned release (same day)

The newest upstream tag is still the pinned 2.2.0, but upstream's main branch has moved 119 commits (tip of 24 November). Read-only pin triage against that tip: verdict "patches affected"; no drift failures and no coupling failures; plugin ABI 43; 327 changed files; all 48 patch target files and six rewrite-anchor files (the all-reduce entry, the fp8 fused quant, the 8a8 prefill header, the fp8 GEMM, the rows file, the selftest) changed upstream. Whether the patch set still applies is the next check (the composition gate against that tree).

What changed for gfx11, which supersedes the first-part finding above:
- Upstream's native RDNA3 library gained the int8 trunk GEMM at both the plain int8 form and the form with rotated activations (the form of our IQ8R container): one row for every M, wave-matrix walk, exact integer group sums, a 128x128 tile from M of 128 up, with bit-for-bit selftests.
- On 24 November it gained an exact packed dense IQR GEMM on gfx1100 (a GEMM unit, a layout unit and device tests).
- Also new in the 17 units it changed: rope and rmsnorm rows, a causal convolution row, paged attention staging and split-KV scratch from geometry.
- libr4d (gfx12) gained a bf16 GEMM over bf16x12 weights, a wide scalar head for M above 8, an f32 WMMA logits row for 17 to 64 rows, head_dim 128 attention at gqa 3, and model-specific operators.

Consequence: the "no native row for IQR or the int8 form" statement holds for the pinned release only. At upstream main there is a native row for each, so part two of this item (matched arms on one card) must include them: upstream's native library at main as arm B against our libr3 on the IQR and IQ8R rows, decode and prefill, with output comparison. The decision to bump follows a tagged release; until then main is read for evidence only.
