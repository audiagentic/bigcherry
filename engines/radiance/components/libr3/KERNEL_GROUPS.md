## Kernel-family development builds

The gfx1100 kernel implementation is shared across models. No model-specific
kernel binaries or copies of upstream sources are created. `libr3` remains
the single canonical plugin, containing all supported upstream units and the
native/compatibility rewrites selected at configure time.

With a Ninja component build configured as above, compile only the HIP units
of a family during development:

```sh
cmake --build BUILD_DIR --target r3_group_gemm_quant -j 1
cmake --build BUILD_DIR --target r3_group_attention -j 1
cmake --build BUILD_DIR --target r3_group_speculation -j 1
```

The eight `r3_group_*` targets are `core`, `gemm_quant`,
`gemm_dense`, `attention`, `moe`, `speculation`, `multimodal` and
`collectives`. `r3_groups_all` compiles all upstream HIP units but does
not link the plugin. The existing `libr3` target compiles *and links*
everything. Family targets depend on the *same object files* as `libr3`;
running a full build afterward reuses completed compilation.

`BUILD_DIR/r3_groups.tsv` records exactly one build-family owner per
upstream translation unit. CTest's `r3_groups_test` verifies that every
upstream unit is covered exactly once and that key kernels remain assigned
to their expected families. A new upstream unit defaults to `core` until it
is classified explicitly; review the manifest at every pin bump.

Models and engine rows **do not define compilation boundaries**: a model
may use several families, and one family can support many models. For
example, DFlash2 invokes shared IQR GEMMs from `gemm_quant` and its own
operators from `speculation`. A video model uses `multimodal`, as well
as shared dense GEMMs and attention. Build these families as required, but
do not create model-specific copies of a kernel or encode model names into
build flags.

**Qualification:** a family build only proves compilation of those objects.
It does not produce a loadable plugin, and it cannot certify the family in
isolation. Before any release, run the full `libr3` link, its component
tests, relevant `r3_selftest` rows and downstream engine qualification.
Keep `R3_ONLY` empty for that full build; nonempty `R3_ONLY` is still
supported for narrow experiments but is not a release build. Native rewrite
coverage and shimmed compilation remain enforced by the full build.
