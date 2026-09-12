# 1240_hi119_moe_glu_file_cli

Plan item HI119 (superseded by THA02, `tuning-hip-autotune`). See
`SUMMARY.md` for the mechanism.

## Status

Same schema-2 gap as `1238`/`1239`: this patch adds the `--moe-glu-file`
CLI hook that lets a Python evidence producer instantiate patch `1239`'s
test case from an observed dispatch shape at runtime. It was exercised on
real hardware during HI119's original session, but that evidence predates
the schema-2 ABI bump and was deliberately not carried forward as
validating STATE.

Depends on patches `1238` and `1239`.

## Disposition

`state` stays `"untested"`. `kind = "diagnostic"`. THA02
(`docs/planning/active/tuning-hip-autotune/THA02.md`) owns the fresh
schema-2 hardware pass this needs before any promotion.
