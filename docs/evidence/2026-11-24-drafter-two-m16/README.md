# VTP37 — IQR M17–64 NT1 slicing (gfx1100)

### Stage 1 — full native build and resource census
- Full native libr3 plugin built and linked: `R3_NATIVE=ON`, `R3_ONLY` empty, no excluded units (115 included).
- One named 16-token tile constant shared by the device kernel, launcher and sliced dispatch.
- New route limited to rotated `i8a8h` B4-table BF16-scale M17–64; other forms and upstream gfx1201 unchanged.
- Stock/rewrite IQR code-object metadata: 840 signatures, zero added/removed/changed resource tuples.
- AQL chained-slice accounting backed by per-kernel packet semantics in source; live AQL tape playback not tested.


### Stage 2 — FP64 qualification
- Final built, uninstrumented plugin: M17–64, seven shapes and 336 whole-operation GPU comparisons; 115,489,152 BF16 outputs per route.
- Stock-vs-new independent FP64 numerical gate: PASS 7/7 on max absolute error and RMS-relative tolerance 1e-6; ULP recorded but not gated.
- M17/M64 launch-count tests demonstrate two/four ordinary NT1 slices for plain and chained kernel specializations; no live AQL chain tape validation.

### Stage 3 — cold paired replay
- Same-session stock/new: 256 MiB cold wash, three repetitions, 7 shapes x 12 heights x 2 specializations.
- New route faster for every measured M17–64 on every shape in both specializations; no stock-faster crossover.
- Five-layer M24 unchained total: 18.3088 -> 4.7199 ms (3.88x).
- Five-layer M24 chained-kernel total: 18.4291 -> 4.4641 ms (4.13x).
- Five-layer M32 unchained total: 18.4778 -> 4.7292 ms (3.91x).
- Five-layer M32 chained-kernel total: 18.6023 -> 4.5097 ms (4.12x).
- Only synthetic kernel timing, no model serving, KLD, A/B, or live AQL chain tape.

### Stage 4 — final IQR selftests and review
- `gemm_iqr_nt_m64`: 1034 OK, 0 FAIL (includes M16, M17 and M64).
- `gemm_iqr_tiled`: 129 OK, 0 FAIL.
- `gemm_iqr_gated`: 36 OK, 0 FAIL.
- Plain/chained M17/M64 packet count and output equality verified under HIP. Live AQL completion is source-backed but not runtime-tested.
- Ready for owner review, not shipped. No serving, A/B or KLD.

### Final consistency review
- Rejected an intermediate partial relink that emitted one stock-M64 packet at M17/M64; the numerical pass alone was insufficient.
- Rebuilt all four IQR parts and relinked cleanly without instrumentation; final M17/M64 launch checks passed with 2/4 NT1 dispatches. Only then reran the FP64 and cold-replay gates. Root cause of intermediate partial-relink inconsistency was not isolated.
- Full four-part matched-stock resource census: p0 740, p1 796, p2 840, p3 796 = 3,172 signatures; zero VGPR, spilled-VGPR or private-byte differences.
