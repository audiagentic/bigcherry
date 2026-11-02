# 1240_hi119_moe_glu_file_cli

## Summary

Adds a --moe-glu-file argument (a new moe_glu_file_path parameter on test_backend(), independent of --test-file) that lets a Python evidence producer instantiate patch 1239's test_bigcherry_moe_glu_fusion class with runtime-derived parameters via a one-line `type glu_op k n m n_mats n_used broadcast` format, without recompiling test-backend-ops.

## Outcome

Recorded state: `untested` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

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

## Real schema-2 hardware pass (2026-09-12, dev dual gfx1100, THA02)

Built and included in the same real hardware pass as `1236`/`1238`/`1239`
(full chain compiles and runs cleanly on both real XTX GPUs). The
`--moe-glu-file` CLI hook itself was not separately exercised this pass
(the correctness net used the statically-registered instances from `1239`,
not the CLI-driven path); confirming the CLI hook against a real observed
dispatch shape is still open before promotion. Full detail in THA02's plan
item.

## Disposition

`state` stays `"untested"`. `kind = "diagnostic"`. Compiles and coexists
cleanly with the rest of the chain on gfx1100 as of 2026-09-12; the CLI
hook's own dedicated exercise against a real observed dispatch shape is
still open before any promotion.
