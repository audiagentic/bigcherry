# rad0009_serving_manifest

**Status:** untested
**Plan item:** RSC07

## Summary

Lets the engine start from a resolved serving manifest: `engine --manifest PATH [--role engine] [-v|-vv]`, with no command-line converter in our tooling.

## Outcome

untested: package and card-free tests exist; a manifest launch is not yet shown on hardware to parse to the same config as the argv launch.

## Detail

See below.

## What it does

`core/bc_serving_manifest.{h,cpp}` (created by the patch) hold all the logic; `core/main.cpp` gains one include and two calls, nothing else is touched (`config.cpp` and its option table are not edited, so `--manifest` is not listed in `--help`).

- Before `config_parse`, `bc_serving_engine_argv` replaces argv. With `--manifest` only `--role engine`, `-v` and `-vv` may accompany it; any other flag is an error. The manifest must have `schema_version` 1, a `values` object, `hash_algorithm` "sha256" and a `content_sha256`.
- Every `engine.<name>` value becomes `--<name with _ as ->`, in sorted key order (valued options first, then bare flags), and radiance's own parser and validation then run on it. An unknown key (not in `rad::diag::config_option_names()`) is an error naming it. The argv shape follows the JSON type alone: `true` adds a bare flag, `false` omits it (the default stands), a string adds `--flag value`, an integer or finite float is formatted exactly. Null, arrays, objects and non-finite numbers are errors. A shape the option does not accept (a value for a boolean option, `true` for an option needing an argument) is left to `config_parse`, which rejects it, and `main` exits 2.
- `engine.radiance_home` and `engine.kernels` must be present and non-empty, so `RADIANCE_HOME` and `RADIANCE_KERNELS` (which `config_parse` consults when the option is empty) cannot silently supply them.
- After a successful parse `bc_serving_mark_given` checks that every option it produced was parsed as given (else start-up fails with exit 2, e.g. a bare flag that swallowed the next one) and relabels those sources as `manifest <first 12 chars of content_sha256>` in the startup report.
- Errors print a message and `main` exits 2.

The manifest's content hash is not recomputed by the engine; `bc-serve` produces it.

## Why a patch

The engine is upstream code and reads its configuration only from the command line; the owner decision (RSC01, 2026-11-02) is that each process loads its own manifest through radiance's own option table rather than a converter generating flags.
