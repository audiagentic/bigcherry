# Archived Radiance patches

Retired patches are kept here unchanged, as evidence of what was tried and what worked or failed (rule:
`docs/reference/patches/PATCH_SYSTEM.md`, "Retired patches: archive, never delete"). Nothing under `archive/` is ever
discovered, composed, linted or built: the registry, `patch-lint`, `pin-bump`, recipes and builds skip it
(`bigcherry.patch.registry.ARCHIVE_DIR`, tested in `tools/tests/patch/test_patch_registry.py`). An archived patch is
not revived in place; reusing the idea means a new patch identity.

Each package is moved whole and unmodified (`patch.toml`, code, README/SUMMARY): its state, content and digest are what
they were when it was retired. One row per patch; the transition is also an unreleased ledger event.

| patch id | final state | archived | reason | successor | evidence |
|---|---|---|---|---|---|
| rad0001_drain_mover_before_arena_free | superseded | 2026-11-20 | Fixed a real two-XTX teardown hang (3/3 clean exits against 2/3 hangs unpatched); later teardown work replaced it. Plan RAD01. | later teardown work in the engine (none carried as a patch) | `rad0001_drain_mover_before_arena_free/SUMMARY.md` |
| rad0006_heat_freeze_diagnostic | superseded | 2026-11-20 | Diagnostic build only (`RAD_DIAG_FREEZE_PROMOTION_AFTER=N`); its question is closed: freezing promotions did not restore Qwen3.6-35B-A3B MTP acceptance, so the mover is ruled out. Plan RAD14. | none | `rad0006_heat_freeze_diagnostic/SUMMARY.md` |
| rad0034_prefill_concurrency | superseded | 2026-11-20 | Prefill admission cap (`--max-prefill-seqs`), untested and never promoted; replaced by the prefill governor. Plan VTP39. | rad0036_prefill_governor | `rad0034_prefill_concurrency/README.md`, `SUMMARY.md` |
| rad0035_prefill_order | superseded | 2026-11-20 | Prefill ordering and long-prefill threshold on top of rad0034, untested and never promoted; replaced by the prefill governor. Plan VTP41. | rad0036_prefill_governor | `rad0035_prefill_order/README.md`, `SUMMARY.md` |
