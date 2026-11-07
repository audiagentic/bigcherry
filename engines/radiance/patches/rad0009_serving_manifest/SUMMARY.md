# rad0009_serving_manifest

**Status:** untested
**Plan item:** RSC07

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
