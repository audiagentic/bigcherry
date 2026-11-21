# rad0009_serving_manifest

**Status:** validated
**Plan item:** RSC07

## Edition 2 projection removed (RUM04, 2026-11-21)

The seam described in the next section is gone. Edition 3 is the only wire edition, so the patched
`bc_serving_engine_argv` has no `kContractVersion` fork: every start admits the sealed native YAML snapshot
(`--role engine`) or a measuring manifest file, and the engine's options come from that YAML. Removed with it:
`v2/native_option_projection.h` and its test `bc_v2_native_projection_test`, the registry-to-flag projection, the
`bc_serving_manifest_host_pool_total` aggregate conversion with its `BC_RAD_HOST_POOL_MIB_AGGREGATE` compile
definition, and the `g_emitted` given-check that only the projection fed. The history below and the evidence
files describe the earlier design and are left as written.

## Current v2 projection boundary (RUM04 migration seam, 2026-11-18)

The v1 description below records this patch's original design; the production
patch now admits the **strict serving-v2 manifest**. Its single source of
upstream option and environment names is
`components/serving/contract/serving.toml`, generated as
`v2::kFields` (113 native option bindings) and `v2::kEnvBindings`
(15 environment bindings). No second field-to-flag table is maintained here.

`bc_serving_engine_argv` in the patched `core/bc_serving_manifest.cpp`
now calls the owned C++17 header
`components/serving/v2/native_option_projection.h` to project the
generated bindings and emit `projected.argv()`. The runtime checks
every projected name against Radiance's actual option inventory; a changed
option name fails closed until both sides agree. The shared CTest
`bc_v2_native_projection_test` exercises every binding, no-value switches,
and a one-entry port rename that changes emitted argv; the patch package's
card-free test compiles and runs the real patched entry point.

This temporary translation seam is what **RUM04** removes in favor of
the immutable admitted original YAML bytes and upstream fleet parser,
while retaining BigCherry's measured admission and local owner settings.
It is not evidence of 2.x compatibility or a request to move the pin.

## Summary

Lets the engine start from a serving manifest v1: `engine --manifest PATH [--role engine] [-v|-vv]`, with no command-line converter in our tooling.

## Outcome

untested: package and card-free tests exist (the patch's own code is compiled with the shared v1 library under radiance's warning flags and run against the design example); a manifest launch is not yet shown on hardware to parse to the same config as the argv launch (RSC08 S2c).

## Detail

See below.

## What it does

`core/bc_serving_manifest.{h,cpp}` (created by the patch) hold all the logic; `core/main.cpp` gains one include and two calls; `core/CMakeLists.txt` links the shared `bc_serving_v1` library into `rad_core` (failing closed when the lane did not pass its `.cmake`). `config.cpp` and its option table are not edited, so `--manifest` is not listed in `--help`.

- Before `config_parse`, `bc_serving_engine_argv` replaces argv. With `--manifest` only `--role engine`, `--plan-memory FILE`, `-v`, `-vv` and the quality-run flags `--kld-record DIR`, `--kld-corpus FILE`, `--kld-ref DIR` and `--kld-out FILE` may accompany it (the KL mode is a kind of run, not configuration, so it takes no manifest field and keeps its label "command line"); any other flag is an error. `--plan-memory FILE` is the measuring start (rad0011): it is the one flag that makes the engine take an unmeasured manifest (`plan.state` `unmeasured`), and without it an unmeasured manifest is refused (`BC_CFG_PLAN_MISMATCH`).
- The file is read by the shared strict reader (`bc::serving::v1::read_manifest`, naming its plan use: serving or measuring): size, UTF-8, duplicate keys, integers, contract identity, `metadata.content_digest`, the generated structural decode, provenance coverage and every semantic rule, with bc-serve's own errors. The engine does not parse the document itself.
- Every registry field bound to a radiance option (`radiance_option` in `serving-v1.toml`, 19 fields: model, sessions, context, host pool/KV/reserve, checkpoint slots, runtime home and kernels, port, tp, batched tokens, checkpoint interval, placement, n-gram placement, KV dtype, speculative tokens, rope scaling, prefix resume) becomes `--<flag> <value>` in registry order, and radiance's own parser and validation then run on it. A field the manifest leaves out is left to radiance's default. Strings and integers are written as they are, an array of strings is joined with commas.
- `--radiance-home` and `--kernels` must be present and non-empty, so `RADIANCE_HOME` and `RADIANCE_KERNELS` cannot silently supply them.
- After a successful parse `bc_serving_mark_given` checks that every option it produced was parsed as given (else start-up fails with exit 2) and relabels those sources as `manifest <first 12 hex of metadata.content_digest>` in the startup report.
- Errors print a message and `main` exits 2.

The interim bc-serve format (`schema_version`, `values`, `content_sha256`) is gone: it is refused like any other non-v1 file.

## Why a patch

The engine is upstream code and reads its configuration only from the command line; the owner decision (RSC01, 2026-11-02) is that each process loads its own manifest through radiance's own option table rather than a converter generating flags.
