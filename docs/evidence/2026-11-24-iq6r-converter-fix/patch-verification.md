# XDP10 — conditional converter ISA target patch

## Stage 2A — patch and code-object qualification

`rad0062_iq6r_converter_gpu_arch` is an anchored managed patch to
`libquant/CMakeLists.txt`. It adds the optional
`RAD_QUANT_GPU_TARGETS` cache variable (default `RAD_GPU_TARGETS`).
Only `libquant`'s HIP_ARCHITECTURES property is overridden; the
serving and arch plugin targets keep their existing architecture
selection. No trellis arithmetic, data layout or CPU fallbacks change.

Card-free unit checks: anchor application, idempotence, missing-anchor
fail-closed detection. **2 of 2 pass**. Repository-wide patch-lint:
**pass** (pre-existing warnings remain).

A release-mode scratch quantizer build with serving
`RAD_GPU_TARGETS=gfx1201` and independent
`RAD_QUANT_GPU_TARGETS=gfx1100;gfx1201` passed compilation and link.
The generated HIP compiler invocation includes both
`--offload-arch=gfx1100` and `--offload-arch=gfx1201` for
`lq_tcq_gpu.hip`; the core serving HIP compile invocation still
targets gfx1201. Enumeration of the completed quantizer fat binary
reports one host bundle and both gfx1100 and gfx1201 HIP bundles.
This is a positive **packaging** test, not a device-level TCQ
numerical or full-model conversion acceptance test.

The patch remains `untested` for production. New GPU conversion,
ASAN/UBSan, 27B artifact, full repository patch suite and model
declaration are separate gates, none inferred to pass from this build.

## Full repository patch-suite attempt

The requested full `python3 -m unittest discover -s tools/tests/patch -q`
was executed (1,490 tests). It **did not pass**: 4 failures, 40 errors,
33 skipped. Investigated errors include executables generated under a
non-executable temporary directory (`PermissionError`), missing
`pytest` in the test interpreter, and a separate pre-existing
`NameError` in generic campaign evidence binding
(`write_json_atomic`). The new patch's two focused tests still pass.
Do not count the full suite as qualified or promote the patch based on
this attempt. These baseline issues were not modified in this slice.
