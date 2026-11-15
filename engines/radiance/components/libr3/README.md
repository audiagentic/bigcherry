# libr3: gfx1100 kernel plugin for radiance

## Summary

BigCherry's gfx1100 (RX 7900 XTX) kernel plugin for radiance (RAD01 PR 5): native
WMMA forms for the GEMMs, attention and all-reduce, built against the pinned
engine's libr4d sources via `rewrites.txt` + `native/*.rw` and loaded through
`RADIANCE_HOME`. It is BigCherry source compiled against the installed engine,
not a patch to the engine itself.

## Outcome

Native gfx11 attention defaults on through the typed serving manifest's
`spec.processes.engine.kernels.attn_native` setting. The former
`R3_ATTN_NATIVE` environment switch is no longer read; see the
[current model pages](../../../../docs/reference/engines/radiance/README.md#current-pin-throughput)
for measured throughput. The setting selects the native form for both the plain and the KV-split prefill
launches (the split's partial store takes the accumulator converted to the gfx12 layout; RAD52). The native
prefill attention runs without the late K prefetch (`O_KLATE`), which spilled the fp8 and k6_v4 kernels, and 8-bit
KV widens to f16 through the exact direct conversion in `r3_e4m3_f16.h` (RAD53). Evidence:
`docs/evidence/2026-11-15-rad52-native-split-store/` and `docs/evidence/2026-11-15-rad53-fp8-prefill-attention/`.

## Lab tools

`r3_selftest <libr3.so>` (target `r3_selftest`) checks and times kernels on one card; it needs no engine or model.
An out-of-tree build of a branch's component (`cmake -S engines/radiance/components/libr3 -DRADIANCE_SRC=<source>
-DCMAKE_PREFIX_PATH=<installed radiance>/runtime-prefix -DR3_TARGETS=gfx1100 -DR3_NATIVE=ON`) gives both the selftest
and a `radiance_home/kernels/libr3.so` an engine can load first through `--radiance-home NEW:STOCK`.

- `--case <row>`: correctness rows for one kernel row name (e.g. `attn_prefill_h256_gqa6_fp8kv`,
  `attn_prefill_split`); count `ok` and `FAIL` lines over the whole output.
- `--perfpaged Q CTX [iters]`: paged prefill attention, all four KV formats, Q query rows at the end of CTX keys;
  `--perfpaged-kv N` sets the KV heads (default 2, a TP2 rank; 1 is a native TP3 rank).
- `--perfw8 N K M [iters]`: the 8-bit trunk GEMMs at one shape; `--cold MIB` reads MIB through the caches before every
  timed launch (256 empties the card's L2 and last-level cache: weights arrive cold, as in decode); `--pin-mw N` / `--pin-nt N` pin the narrow GEMM's wave and token-tile counts for a sweep.
- Register pressure and spill per instantiation: compile one unit with
  `--offload-device-only -Rpass-analysis=kernel-resource-usage` (VGPRs, `ScratchSize`, occupancy per template).

## Detail

See `component.toml` for remaining environment-only lab/selftest settings and the
serving manifest for engine kernel switches (status vocabulary:
`production`/`experimental`/`evaluated`/`diagnostic`/`tuning`) and
`docs/archive/radiance/PLATFORM_DESIGN.md` decision 2 for why this is
a component rather than a patch. Native rewrite rules are written against
radiance v1.14.0's libr4d and refuse to apply against any other revision
(`built-against`); `R3_NATIVE=OFF` falls back to building every kernel through
`r3_compat.h` only. Evidence for each setting lives in the dated entries of
`docs/archive/radiance/LAB_HISTORY.md` (the lab log).
