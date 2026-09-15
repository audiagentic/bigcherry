"""PA37 fallback validators for 1203's plural-contract validation plan.

Every check in ../validation.toml is normally producer-supplied by
producer.py's "rd050607" producer (see producer.toml), dispatched through
``execute_validation_producer()``. These callables are the framework-
required ``evaluate_check()`` fallback, reached only if the plan is ever
evaluated WITHOUT ``--validation-producer 1203_rd050607_rdna4_wmma_fa_q6k_mmq/
rd050607`` (e.g. a bare plan-resolution/lint pass). They deliberately
return FAIL, not a fabricated PASS -- this patch's correctness/performance
claims must never be satisfied by a path that did not actually run the
real producer (mirrors the project's fail-closed doctrine; unlike PA36-F's
synthetic fixture, whose contract-scoped checks are non-load-bearing demo
content, 1203's checks gate real RD05/RD06/RD07 evidence).
"""

from __future__ import annotations

from bigcherry.patch import validation as pv

_FALLBACK_SUMMARY = (
    "fallback: this check requires the 1203/rd050607 validation producer "
    "(--validation-producer 1203_rd050607_rdna4_wmma_fa_q6k_mmq/rd050607); "
    "evaluate_check() never independently satisfies RD05/RD06/RD07 evidence"
)


def _fallback(check_id: str, capability: str) -> pv.ValidationResult:
    return pv.ValidationResult(
        check_id=check_id, capability=capability, status=pv.FAIL,
        summary=_FALLBACK_SUMMARY,
    )


def rd05_backend_reference(ctx: pv.ValidationContext) -> pv.ValidationResult:
    return _fallback("rd05-backend-reference", "correctness")


def rd06_backend_reference(ctx: pv.ValidationContext) -> pv.ValidationResult:
    return _fallback("rd06-backend-reference", "correctness")


def rd06_performance(ctx: pv.ValidationContext) -> pv.ValidationResult:
    return _fallback("rd06-performance", "performance")


def rd07_backend_reference(ctx: pv.ValidationContext) -> pv.ValidationResult:
    return _fallback("rd07-backend-reference", "correctness")


def rd07_performance(ctx: pv.ValidationContext) -> pv.ValidationResult:
    return _fallback("rd07-performance", "performance")
