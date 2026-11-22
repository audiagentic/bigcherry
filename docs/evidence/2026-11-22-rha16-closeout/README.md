# RHA16 — current-main lifecycle advisory revalidation

Date: 2026-11-22. Baseline: BigCherry main at `d226e83d6`.
Disposition: **completed as a bounded verification, with zero new implementation**.
The historical `docs/planning/completed/run-hip-autotune/RHA07.md` remains
the accepted implementation. RHA16 existed only to resolve an erroneous
second active RHA07 identity and to check for residual integration gaps.

## Current source boundary inventory

| Lifecycle | Existing result owner and call site | Existing shared evaluator / artifact | Verification |
| --- | --- | --- | --- |
| Build | `tools/bigcherry/cli/build.py:188-213` | `campaign/run_advisories.py:170`; `build-advisories/<run-id>/advisories.json` | `test_run_advisories.py:72-76` |
| Paired A/B | `tools/bigcherry/campaign/benchmark.py:1156-1162` | `campaign/run_advisories.py:128`; sibling `advisories.json` | `test_run_advisories.py:50-69` |
| Recovery | `tools/bigcherry/tuning/workflow.py:1025-1049` | `campaign/run_advisories.py:199`; sibling `recovery-advisories.json` | `test_run_advisories.py:78-86` and `test_recovery.py` |
| Promotion | `tools/bigcherry/tuning/tune_promotion.py:824-840` | `campaign/run_advisories.py:229`; sibling `promotion-advisories.json` | `test_run_advisories.py:87-91` and `test_tune_promotion.py` |

All four use the one `RunEvaluation`/stable schema in
`tools/bigcherry/campaign/run_advisories.py:51-67`. No alternate advisory
evaluator, worker, schema, dispatcher or acceptance path is required.

## Actual regression run

```text
PYTHONPATH=tools /opt/venv/bin/python -m pytest \
  tools/tests/campaign/test_run_advisories.py \
  tools/tests/tuning/test_workflow.py \
  tools/tests/tuning/test_recovery.py \
  tools/tests/tuning/test_tune_promotion.py -q

130 passed, 19 subtests passed in 8.28s
```

No GPU runs, code changes or production throughput claims. The advisory
evaluator only returns immutable `RunEvaluation` objects; A/B writes its
artifact separately from the authoritative `run.json`; build exit policy
continues to inspect worker results rather than advisory findings; recovery
advice is written after the decision and before the preexisting failure
exception; promotion returns its original result. Negative fixtures cover
missing evidence/degraded/failure/unknown without changing child state.

## Decision

No residual missing lifecycle adapter was found on current main. RHA16
can be **completed** as the audit-only follow-up, referring to completed
RHA07; leave the historical RHA07 unchanged. Any genuinely new future
boundary defect requires a source-specific failing regression and
existing owner, not a second generic RHA implementation.
