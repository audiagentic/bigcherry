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
for measured throughput.

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
