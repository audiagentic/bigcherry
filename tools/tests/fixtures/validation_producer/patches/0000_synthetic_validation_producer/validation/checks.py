"""PA36-F step 7 fixture: the ONE unscoped/universal check this synthetic
patch declares, deliberately left for the shared evaluate_check() fallback
(never producer-supplied) -- proves execute_validation_producer()'s step 9
("producer-supplied result, else evaluate_check()") for real, not just the
producer-supplied path."""

from __future__ import annotations

from bigcherry.patch import validation as pv


def universal_check(ctx: pv.ValidationContext) -> pv.ValidationResult:
    return pv.ValidationResult(
        check_id="syn-universal-smoke", capability="smoke",
        status=pv.PASS, summary="synthetic universal smoke check always passes",
    )


def check(ctx: pv.ValidationContext) -> pv.ValidationResult:
    # Not exercised by the PA36-F step-7 end-to-end test (both contract-
    # scoped checks are producer-supplied there), kept only so this
    # fixture's validation.toml is fully self-consistent/loadable on its
    # own by any future direct evaluate_check() exercise.
    return pv.ValidationResult(
        check_id="syn-c1-correctness", capability="correctness",
        status=pv.PASS, summary="fallback (should not normally be reached)",
    )
