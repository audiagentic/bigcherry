# 1320_meta_compute_timing

**Status:** validated
**Plan item:** QFP16

## What it does

Diagnostic only. With `BIGCHERRY_SUBMIT_TIMING=1` every meta-backend (-sm tensor) graph_compute logs
`BIGCHERRY_META_TIMING n_nodes n_subgraphs rebuild rebuild_us launch_us allreduce_us total_us`: subgraph rebuild cost,
the summed per-subgraph per-device async launch host time, and the summed AllReduce enqueue host time. Attributes the
~5 ms target verify submit (1319) to subgraph count x device launches vs AllReduce enqueue vs rebuilds.
