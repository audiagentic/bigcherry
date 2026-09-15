# 0810_replay_hit_diagnostics framework validation

Adds the opt-in `GGML_HIP_REPLAY_DIAGNOSTICS` build option and the hit
recorder/aggregated-JSONL-log plumbing it gates (via
`GGML_HIP_DISPATCH_HIT_LOG`). Production replay builds do not enable this
option, so the normal replay lookup path compiles with no diagnostics branch
and no synchronization cost; a diagnostic replay build gets full per-dispatch
hit-log visibility instead. PA26's own hardware-run harness (see
`tools/bigcherry/campaign/replay_equivalence_hardware.py`) uses this option
to build a DIAGNOSTIC companion arm alongside the diagnostics-free production
arms it actually compares.

This local framework adapter has no Experiment Contract: it checks build
plumbing, not a claimed kernel speedup. Historical `validated` state is not
current qualification.

Validation is the universal `apply`/`build` checks only -- no patch-specific
custom check exists for this package. Passing proves the patch applies
cleanly to the pinned upstream source and the resulting tree compiles under
both the default (diagnostics-off) and the diagnostics-enabled option; it
does NOT prove the emitted hit log is well-formed or attributes hits
correctly at runtime.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework.
