# 1100_hi70_direct_op_evidence framework validation

Adds a small, deterministic `test_mul_mat` corpus to `test-backend-ops` --
four M=127,K=256,N=128 quant cases (MMQ fallback: row count not a multiple
of 128) plus an exhaustive F16 N=1..16 sweep (MMF narrow-batch widths) --
covering candidates that are hard to reach and not reliably exercised by
ordinary model-dependent workloads. RE15 stage F's replay-full generate
requires `status='ok'` correctness evidence for every enumerated non-native
candidate; 24 of gfx1201's 100 candidates never got evidence from any
real-model workload this project tried. Some MMQ fallback cases were
eventually reached by hunting for real models with an irregular dense
dimension, but MMF's narrow batch widths specifically require
speculative-decoding multi-token draft-batch verification, which no
ordinary single-stream prompt/decode call ever produces. Running this
deterministic corpus in tune mode is what produces the candidate evidence;
the patch itself only adds the test cases, not the evidence.

This local framework adapter has no Experiment Contract: it checks build
plumbing, not a claimed kernel speedup. Historical `validated` state is not
current qualification.

Validation is the universal `apply`/`build` checks only -- no patch-specific
custom check exists for this package. Passing proves the patch applies
cleanly to the pinned upstream source and the resulting tree compiles (i.e.
`test-backend-ops` still builds with the new corpus cases registered); it
does NOT itself run the corpus or prove the constructed shapes actually
reach the intended fallback/narrow-batch candidates at runtime.

## Upstream / provenance

Local design, part of this project's own correctness-evidence work (HI70),
per gpt-auto-agent's deep-dive recommendation.
