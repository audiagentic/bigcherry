# 1319_process_ubatch_timing

**Status:** validated
**Plan item:** QFP16

## What it does

Diagnostic only. `BIGCHERRY_SUBMIT_TIMING=1` logs one `BIGCHERRY_SUBMIT_TIMING` line per process_ubatch call
(context, n_tokens, graph reused, graph build/reuse us, set_inputs us, graph_compute host us). Splits the 5-7 ms target
verify submit time measured by 1317 into graph, inputs and scheduler/launch time.
