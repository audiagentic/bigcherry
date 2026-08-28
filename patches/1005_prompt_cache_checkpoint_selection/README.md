# 1005 prompt-cache checkpoint selection

This package owns the implementation, metadata, and validation fixture for
the hybrid/recurrent checkpoint-selection backport. The C++ fixture and its
recorded unit-test result are under `validation/fixtures/`.

The patch remains `untested` until the live multi-turn server validation is
run; the fixture validates the reusable-token arithmetic only.
