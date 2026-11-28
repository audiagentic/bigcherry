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

For Radiance 2.3.0 the inventory grows from 115 to 130 upstream units.
Keep the same eight families: GGML GEMMs and int8 logits belong to
`gemm_quant`, BF16x12 to `gemm_dense`, DS41 sparse ops to `attention`,
DSpark walk to `speculation`, position-table resampling to `multimodal`,
and SiTU expert activation to `moe`. DS41 elementwise and Inkling
control fusions remain shared `core` units. The eight new registry/shape
`.inc` fragments are copied and tracked by the shared rewriter; they
are include dependencies, not additional compilation owners.
libr2 uses the same upstream unit parser (including mixed-case names),
but retains its explicitly qualified RDNA2 unit allowlist.

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

### Follow-up: model and affected-family qualification

The existing planner now accepts repeatable `--row`, `--touched`, and a
`--pin-triage` JSON report. The latter imports the existing changed-files
inventory. Local includes provide a lower bound; headers, deletions, registry
changes or unknown paths conservatively widen to the full unit inventory.
This is not a substitute for compiler dependency evidence.

For model-driven selection, save the serving owner's existing Radiance
`/graph` JSON and supply `--model-graph GRAPH.json`. Its declared bands carry
actual selected row names, provider names, geometry, model rank/world and
plugin metadata; the planner retains them and the graph digest, rejects
malformed declaration coverage, and marks unknown or mismatched routes
uncovered. A declared graph is not a physical execution trace; other ranks
and unresolved bands remain gaps. No live server is started by the planner.

Measurement commands now require `--bdf` with a physical PCI address and the
existing named GPU lease. The harness resolves the HIP device and checks the
selftest's actual BDF/architecture, records observed controls, hashes the
build artifacts and family policy, and rejects missing or changed identities.
Explicit composed source is copied into private build scratch before edits.
Replays use the configured shapes in both cold and warm modes; repeated
overlay timings and layout-refusal records must cover every expected hash
case. Duplicate/malformed hashes, skipped required cases, missing timing,
regressions and unconverged replay cannot be a pass.

The existing `gpu-ai-qualify plan --native-plan PLAN.json` imports this scope
into its managed capture, retaining unresolved rows, compatible-reference
requirements and input identities. It never treats independent stability
tests as native or production qualification.

Validation: 115 canonical units and 109 configured units on the retained
gfx1100 build; the stale manifest with six excluded units was refused.
Explicit IQR overlay selection resolved embed and tiled routes, while M64,
gated and an unknown attention row retained gaps. Managed capture
20261126T050013Z-gpu-ai-qualification-plan-0cfc3ad6a040 retained incomplete
native scope. Graph import was tested against the upstream export schema
with fixtures, not a new live model capture. 280 focused tests passed,
one skipped. Physical selftest identity, selected-family cold/warm replay
and matched power/production evidence remain unverified by this change.
No serving kernels, profile settings or release acceptance gates changed.
