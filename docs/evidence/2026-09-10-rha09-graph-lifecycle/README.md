# RHA09 durable graph-lifecycle evidence

This bundle preserves the raw artifacts needed to re-check the historical HI14
dual-XTX graph-capture run without hardware. The run used Qwen3.8-27B-Q8_0,
two Radeon RX 7900 XTX devices (ROCm0/ROCm1), tensor split, and graph lifecycle
tracing. The server log contains all four lifecycle markers and reports 5,706
of 5,706 matmul launches reaching measured dispatch (100%).

## Offline validation

`lifecycle.json` is produced by
`bigcherry.graph_lifecycle_evidence.capture_lifecycle_from_log`. `validator.json`
is accepted by `bigcherry.tuning.multi_gpu.validate_multi_gpu_evidence`; the
per-device counts represent the two-device dispatches in the retained record.

Raw SHA-256 hashes:

- `server.log`: `bb0ffcce0707d4cebda2ab202e81d7ad20de8f2b9308eb8a2c869baa37338d2c`
- `dispatch-record.jsonl`: `4408219da74d5246a12770950665aaf44c66d72e73417849ad0f8ab48ba926eb`

The raw files were copied from the Brutus artifact directory
`/home/audumla/bigcherry/artifacts/hardware/20260823-hi14-graph-lifecycle/`.
The bundle is historical evidence, not a new performance admission; build
metadata and mixed-topology/long-context coverage remain outside this item.
