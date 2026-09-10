# 1255_nro06_adaptive_mtp_depth

Plan `NRO06`; state `untested`; backend-agnostic orchestration draft.

This package introduces a pure `bigcherry_nro06_adaptive_mtp` controller with source policy constants: a depth-dependent full-accept climb threshold and drop pressure `max(depth*5,20)`. It contains `reset()` and `update()` and has no side effects outside its own state.

The controller is intentionally not yet instantiated by `common_speculative_impl_draft_mtp`. A later stage will add the `draft-mtp-adaptive` type, CLI floor, per-sequence controllers, current draft-cap integration, acceptance feedback, and request/reset plumbing after unit tests are committed.

Fixed `draft-mtp` remains unchanged in this draft. This makes the first patch safe to review and permits exhaustive offline tests of the policy before workload behavior changes.
