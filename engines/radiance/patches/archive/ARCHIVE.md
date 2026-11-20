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
| rad0003_mtp_full_head_switch | rejected | 2026-11-20 | Tested and not accepted: `RAD_MTP_FULL_HEAD=1` (draft rounds through the trunk LM head) gave MTP acceptance 0.63-0.65 against 0.62-0.64 with the container draft head on the measured Flash-Next workload, so draft-head precision does not explain the acceptance gap. Plan RAD01. | none | `rad0003_mtp_full_head_switch/SUMMARY.md` |
| rad0007_hc_pf2_fp8_down | rejected | 2026-11-20 | Tested and not accepted: after the P0 fix and four optimisation passes the FP8 down path takes 1203.7 us at M=2048 on gfx1201 against the existing bf16 path's 530.7 us (about 2.3x slower). Plans RAD15 and RAD17 (RAD15 deprecated with a pointer here, so the attempt is not repeated blindly). | none | `rad0007_hc_pf2_fp8_down/SUMMARY.md` |
| rad0039_moe_heat_observer | untested (abandoned) | 2026-11-20 | Abandoned: RHP02 slice A only, an inactive read-only credit observer ABI that nothing registers. The engine-side consumer (slice B) was never built, so it never had an activation or a hardware result. Its host ABI test (`heat_observer_abi_test.cpp`) and package test (`test_radiance_rad0039.py`) moved with it to `rad0039_moe_heat_observer/tests/` (not built or run: the archive is never loaded). Plan RHP02 is deprecated. | none | `rad0039_moe_heat_observer/SUMMARY.md` |
