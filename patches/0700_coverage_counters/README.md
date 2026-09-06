# 0700 framework validation

HI168 owns this package's diagnostic-isolation review. This local framework
adapter has no Experiment Contract: it checks emitted instrumentation, not a
claimed kernel speedup. Historical `validated` state is not current qualification.

`family-hook-isolation` upgrades the five actual emitted family-hook fixtures
and checks idempotence. It compiles and runs OFF/ON host C++ controls: OFF must
link without diagnostic function definitions and retain functional dispatch;
ON must execute the probes and counters. Reports identify these as host fixture
checks, not completed HIP binary or GPU architecture evidence.

The universal apply and build checks remain required. Missing source/build
evidence keeps the full adapter ineligible even if the fixture check passes.
Real HIP OFF/ON artifacts and identity-bound qualification publication remain
pending the framework execution/evidence integration in HI168/RV140. Diagnostic
coverage is not same-cell tuned-launch proof for an uninstrumented timing run.

Focused offline check: `PYTHONPATH=tools python -m pytest
tools/tests/patch/test_coverage_validation_adapter.py -q` (one shell command).
Raw reports belong in the run directory under `artifacts/patch-validation/`;
only the maintained evidence writer may publish compact qualification records.
No timing from these checks answers HI168's separate server comparison.
