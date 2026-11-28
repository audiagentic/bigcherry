# libr3: gfx1100 kernel plugin for radiance

## Summary

BigCherry's gfx1100 (RX 7900 XTX) kernel plugin for radiance (RAD01 PR 5): native
WMMA forms for the GEMMs, attention and all-reduce, built against the pinned
engine's libr4d sources via `rewrites.txt` + `native/*.rw` and loaded through
`RADIANCE_HOME`. It is BigCherry source compiled against the installed engine,
not a patch to the engine itself.

## Terms

libr4d is upstream's gfx12 kernel library source; libr3d is upstream's own native RDNA3 library (a different library); libr3 is this component: BigCherry's gfx11 provider, libr4d's units compiled for gfx11 through `r3_compat.h` plus our native rewrites. It is upstream's units built for gfx11, so it keeps its upstream-derived name. Kernels we own live in separate overlay plugins named `libbc-<family>-<arch>.so` (`bc` marks what we own), placed first in the kernel hierarchy. Full definitions, the plan items and the ownership rules: `docs/design/radiance-kernel-composition.md`.

## Outcome

Native gfx11 attention defaults on through the typed serving manifest's
`spec.processes.engine.kernels.attn_native` setting. The former
`R3_ATTN_NATIVE` environment switch is no longer read; see the
[current model pages](../../../../docs/reference/engines/radiance/README.md#current-pin-throughput)
for measured throughput. The setting selects the native form for both the plain and the KV-split prefill
launches. GQA6 supplies bf16 and int8 KV rows. Both convert their native accumulator to the
partial-store layout before a KV split. The int8 row uses upstream's scale-aware cache decoder.
The RAD52 and RAD53 measurements below used the previous pin and its cache formats. The narrow 8-bit GEMM keeps its
unguarded form when only the token count is ragged (17-31 rows: 5-7 decode streams at depth 3), guarding tokens
always (VTP37 step 3a). Evidence: `docs/evidence/2026-11-15-rad52-native-split-store/`,
`docs/evidence/2026-11-15-rad53-fp8-prefill-attention/` and `docs/evidence/2026-11-15-vtp37-narrow-gemm-ragged-m/`.

## Kernel-family compilation

For incremental work, Ninja targets compile shared upstream units by family without
linking partial plugins. See [KERNEL_GROUPS.md](KERNEL_GROUPS.md) for the targets,
ownership manifest, model-profile guidance and full-build qualification rules.

## Lab tools

`r3_selftest <libr3.so>` (target `r3_selftest`) checks and times kernels on one card; it needs no engine or model.
An out-of-tree build of a branch's component (`cmake -S engines/radiance/components/libr3 -DRADIANCE_SRC=<source>
-DCMAKE_PREFIX_PATH=<installed radiance>/runtime-prefix -DR3_TARGETS=gfx1100 -DR3_NATIVE=ON`) gives both the selftest
and a `radiance_home/kernels/libr3.so` an engine can load first through `--radiance-home NEW:STOCK`.

- `--case <row>`: correctness rows for one kernel row name (e.g. `attn_prefill_h256_gqa6_i8`,
  `attn_prefill_split`); count `ok` and `FAIL` lines over the whole output.
- `--perfpaged Q CTX [iters]`: paged prefill attention, bf16 and int8 KV, Q query rows at the end of CTX keys;
  `--perfpaged-kv N` sets the KV heads (default 2, a TP2 rank; 1 is a native TP3 rank).
- `--perfw8 N K M [iters]`: the 8-bit trunk GEMMs at one shape; `--cold MIB` reads MIB through the caches before every
  timed launch (256 empties the card's L2 and last-level cache: weights arrive cold, as in decode).
- Register pressure and spill per instantiation: compile one unit with
  `--offload-device-only -Rpass-analysis=kernel-resource-usage` (VGPRs, `ScratchSize`, occupancy per template).

## Detail

See `component.toml` for remaining environment-only lab/selftest settings and the
serving manifest for engine kernel switches (status vocabulary:
`production`/`experimental`/`evaluated`/`diagnostic`/`tuning`) and
`docs/archive/radiance/PLATFORM_DESIGN.md` decision 2 for why this is
a component rather than a patch. Native rewrite rules check exact source anchors at the revision
declared in `component.toml`. `R3_NATIVE=OFF` compiles through `r3_compat.h` alone.
The selected gfx1100 provider remains BigCherry's libr3. The IQR embedding gather, the IQR vocabulary head and the IQR GEMM family
(with the fused single-band tile) are no longer built here: they are owned by the `bc-iqr` overlay plugin, listed ahead of libr3 and libr4d
in the kernels line of every serving config that serves an IQR container (`exclude.txt` names the three units, their rewrites are in
`native-shelved/`, and the 8 shared prefill-header rewrites stay). RCS15–18 track qualification of
upstream's native libr3d. The grouped canonical MXFP4 MoE row was removed upstream in 2.0,
so its old override is in `native-shelved/`. The dense MXFP4 decode and prefill rows remain active. Evidence for each setting lives in the dated entries of
`docs/archive/radiance/LAB_HISTORY.md` (the lab log).
