# libr3: gfx1100 kernel plugin for radiance

## Summary

BigCherry's gfx1100 (RX 7900 XTX) kernel plugin for radiance (RAD01 PR 5): native
WMMA forms for the GEMMs, attention and all-reduce, built against the pinned
engine's libr4d sources via `rewrites.txt` + `native/*.rw` and loaded through
`RADIANCE_HOME`. It is BigCherry source compiled against the installed engine,
not a patch to the engine itself.

## Outcome

`production`: `R3_ATTN_NATIVE` defaults on with quality and speed evidence; see
`docs/reference/engines/radiance/README.md` for the measurement log and the
per-model pages for current numbers.

## Detail

See `component.toml` for the full runtime-setting table (status vocabulary:
`production`/`experimental`/`evaluated`/`diagnostic`/`tuning`) and
`docs/reference/engines/radiance/PLATFORM_DESIGN.md` decision 2 for why this is
a component rather than a patch. Native rewrite rules are written against
radiance v1.9.0's libr4d and refuse to apply against any other revision
(`built-against`); `R3_NATIVE=OFF` falls back to building every kernel through
`r3_compat.h` only. Evidence for each setting lives in the dated entries of
`engines/radiance/README.md` (the lab log).
