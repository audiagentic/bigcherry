# 0100 framework validation

HI168 owns this package's diagnostic-isolation review. This local framework
adapter has no Experiment Contract: it checks build plumbing, not a claimed
kernel speedup. Historical `validated` state is not current qualification.

`coverage-source-selection` applies the actual package upgrade to its emitted
CMake source-list fixture, checks idempotence, and executes CMake for production,
diagnostic, record and tune selection. The result is host CMake configuration
proof only. It does not establish a completed HIP build, a gfx architecture,
runtime activation, numerical correctness or throughput parity.

The universal apply and build checks remain required. Missing source/build
evidence keeps the full adapter ineligible even if the fixture check passes.
Full HIP production/diagnostic artifact inspection and current-pin evidence
publication remain pending the framework execution/evidence integration in
HI168/RV140. Do not invoke an unrelated RD-specific contract mode or use a
patch-absent source that cannot build as a substitute control.

Focused offline check: `PYTHONPATH=tools python -m pytest
tools/tests/patch/test_cmake_validation_adapter.py -q` (one shell command).
Raw reports belong in the run directory under `artifacts/patch-validation/`;
only the maintained evidence writer may publish compact qualification records.
No timing from these checks answers HI168's separate server comparison.
