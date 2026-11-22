# PRQ01–PRQ05 — close historical Radiance qualification program against 2.2

Date: 2026-11-22. Reviewed BigCherry main at `d226e83d6`.
Scope: **planning-only reconciliation** of the earlier v1.17 qualification
migration. No new Radiance runtime or qualification result is asserted.

| Item | Historical delivery and source-backed current state | Disposition |
| --- | --- | --- |
| PRQ01 | Nov 16 documented 26/26 source patch applications after repairing a drifting vendor checkout. The lasting vendor-pin refusal is in `tools/bigcherry/patch/rebase.py:1347`, and Radiance is now pinned to 2.2.0 in `engines/radiance/recipes.toml:5-8` | Complete historical rebase diagnosis; current 2.2 admission owned by its own RUM/patch tests |
| PRQ02 | Plan outcome cites merged promotion PRs #983–#988 and #992–#993. Their accepted production subset is present in the current 2.2 `serving-core` recipe; original diagnostic-only candidates are not production-promoted | Complete historical evidence-based promotion campaign; new patches have their own owners |
| PRQ03 | Every named outstanding candidate now has an explicit disposition: `rad0003` and `rad0007` rejected (`engines/radiance/patches/archive/ARCHIVE.md:23-24`); `rad0005` superseded by 2.2 FFmpeg 6 upstream (`ARCHIVE.md:18`); `rad0030` and `rad0036` promoted historically and remain in the production recipe; `rad0006` archived superseded; diagnostic `rad0004`/`rad0010` remain out of production | Complete the **historical decision ledger**, not a new hardware/performance qualification |
| PRQ04 | The original experimental-source split was replaced by a strict vendor-pin and production-source state guard (#990); current policy lives in `tools/bigcherry/patch/source_policy.py`, and `tools/tests/patch/test_radiance_source_classes.py:38-135` tests real source admission, nonvalidated rejection and explicitly declared source classes | Complete revised design, do not revive a duplicate experimental fork |
| PRQ05 | State-based admission and diagnostic G5 landed in #995: `registry.py` `SOURCE_ALLOWED_STATES`, `source_policy.py`, `test_diagnostic_promotion.py`. `docs/reference/testing/PATCH_VALIDATION.md:108-127` documents exactly one state-based rule; rad0004/rad0010 diagnostic with activation records, rad0006 archived | Complete delivered lifecycle and documentation contract |

## Current focused offline verification

```text
PYTHONPATH=tools pytest tools/tests/patch/test_radiance_source_classes.py \
  tools/tests/patch/test_diagnostic_promotion.py -q
11 passed, 5 subtests passed in 0.45s
```

The current real recipe is `pinned-version = "2.2.0"`; its source-class
fixture verifies production accepts only validated patch states. The test
makes each nonvalidated production admission fail, refuses missing diagnostic
activation evidence, and preserves the separate compile-time diagnostic
macro. These are offline gates, not a new clean-build or GPU performance
receipt. Historical v1.17 claims remain historical and must not be
re-presented as fresh 2.2 measurements.

## Ownership boundary

Do not reopen PRQ01–PRQ05 to qualify new patches. The current Radiance 2.2
engine, RUM migration and individual patch plans retain all genuinely open
runtime/quality gates (including RUM06 ABI/lifetime and LTX-specific TP3).
Keep archived patches and prior reviews intact. This plan cleanup neither
asserts native GPU parity nor supersedes uncompleted hardware acceptance.
