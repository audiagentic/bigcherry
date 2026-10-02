# PNRO20 patch-composition repro

Plan item: PNRO20
Status: answered
Owner: bigcherry
Question state: answered

## Question

Which earlier patch in the test-backend-ops composition order breaks patch 1253's anchor?

## Inputs

`repro_compose.py [patch_id ...]`: applies the listed patches in composition order to a scratch copy of
the vendored `tests/test-backend-ops.cpp` and reports where 1253's anchor stops matching.

## Outputs

Console report only; the scratch copy lives in a temporary directory and is discarded.

## Runtime

GPU required: no
Real compilation required: no
Mutates canonical BigCherry state: no

## Safety

- Canonical-state mutation: none (scratch copy only).
- Do not import this experiment from `bigcherry` production, tests, or maintained analysis.

## Disposition

Kept as a diagnostic until PNRO20 closes; then delete.
