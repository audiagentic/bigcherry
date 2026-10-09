# 0820_measurement_signature_shapes framework validation

Stores the canonical signature JSON shape alongside each tuning measurement
result, computed once on the cold path (`ggml_hip_signature_json`) and
serialized into the measurement's JSON output. Downstream tooling needs the
canonical shape associated with a measurement -- not just its raw dispatch
dimensions -- to correctly group and replay candidates.

This local framework adapter has no Experiment Contract: it checks build
plumbing, not a claimed kernel speedup. Historical `validated` state is not
current qualification.

Validation is the universal `apply`/`build` checks only -- no patch-specific
custom check exists for this package. Passing proves the patch applies
cleanly to the pinned upstream source and the resulting tree compiles; it
does NOT prove the emitted `canonical_json` field is well-formed or correct
for any given tuning run.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework.
