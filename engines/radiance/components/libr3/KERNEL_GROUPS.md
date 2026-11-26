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

**Run one build command at a time per Ninja build directory.** Concurrent
Ninja processes sharing the same build tree can damage its dependency log,
causing unnecessary recompilation. Use separate build trees for parallel
experiments, and finish family compilations before the full plugin link.

`BUILD_DIR/r3_groups.tsv` records exactly one build-family owner per
upstream translation unit. CTest's `r3_groups_test` verifies that every
upstream unit is covered exactly once and that key kernels remain assigned
to their expected families in a full build. Intentionally partial `R3_ONLY`
configurations omit that coverage test. A new upstream unit defaults to `core` until it
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

### Qualification selection (RCS26–28)

The build's classifier can enumerate the full upstream inventory, including
units excluded from libr3, without configuring a HIP project:

    cmake -DR3_GROUP_SOURCE=COMPOSED_SOURCE -DR3_GROUP_OUTPUT=OUTPUT.tsv -P engines/radiance/components/libr3/r3_groups.cmake

Use the existing discrete harness to resolve a selected build group or named
unit into configured test routes. From the repository root with BC_WORK_ROOT
set, for example:

    python tools/lab/radiance/unit_harness.py plan --tree COMPOSED_SOURCE --manifest BUILD_DIR/r3_groups.tsv --arch gfx1100 --release RELEASE --group gemm_quant

The optional manifest must be a full, matching libr3 build inventory (not a
partial R3_ONLY build). Without it the planner reports source classification
only. Repeat --group or --unit to select additional work. The output is a
diagnostic JSON plan, not a qualification receipt. It names the existing
build and measurement commands, missing unit/row coverage, and any required
explicit compatible overlay reference (--overlay-reference). Exit 1 means
coverage/reference prerequisites remain; exit 2 is an invalid request.
An excluded base row is not available in the installed libr3 merely because
the installed overlay serves it. The compatible reference must be independently
verified; existence of a file does not prove a matched row or layout.

Compile separately using the existing group target. Run the selected measurement
commands only under the existing GPU lease. Skipped, empty, mismatched or
unconverged checks cannot establish a pass; a new output directory preserves
prior evidence. A passing configured check is still not a production or power
profile qualification. See [the design](../../../../docs/design/radiance-kernel-group-qualification.md)
for the remaining campaign, ownership and profile integration.
