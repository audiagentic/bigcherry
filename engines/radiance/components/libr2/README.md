# libr2

radiance's device kernels for RDNA2 (gfx10.3, the RX 6000 series), built from libr4d's own sources the way libr3 builds them for gfx11.
Today it exists for the remote-MoE helper on a gfx1030 card (`r3_moe_worker` loads `libr2.so` there): the grouped expert GEMM and the
activation quantisers (`units.txt`). Measurements and the reasons: `docs/planning/active/patching-radiance-rdna3/RAD23.md`, RCS07.

- `r3_mma16.h`: the 16x16x16 matrix tile for `libr3/r3_compat.h`'s WMMA wrappers, as ds_bpermute plus f32 FMAs (gfx10.3 has no WMMA).
  libr3 has a file of the same name with the WMMA instruction; `r3_compat.h` includes it with `<>`, so whichever component's directory
  comes first on the include path supplies the tile.
- `units.txt`: the libr4d units built. The integer-WMMA GEMMs and the attention units do not compile for gfx10.3 yet.
- `rewrites.txt`: the one source rewrite gfx10.3 needs (`s_wait_storecnt`).
- `r2_registry.cpp`: the plugin's row table (rows whose unit is not built are dropped at load).
- The rewriter, the stub generator and `r3_compat.h` are libr3's, read from `../libr3`.

Build by hand: `cmake -S engines/radiance/components/libr2 -B <build> -G Ninja -DCMAKE_PREFIX_PATH=<installed radiance>
-DRADIANCE_SRC=<radiance source tree>`. No selftest yet: a unit comes onto `units.txt` when it builds and the helper's
`--compare` against the host reference agrees on a gfx1030 card.
