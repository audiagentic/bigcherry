# 0840 adaptive AllReduce testing

## Scope
0840 is the adaptive internal/RCCL dispatcher selected by `--allreduce adaptive` on top of `0860_allreduce_provider_cli`; the legacy environment provider selector is removed and 0840 requires 0860.

## Hardware-free
- `PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --focal-overlay 0840_hybrid_allreduce_dispatch`
- Add/run a patch unit test that applies 0860 then 0840 and asserts 0840 removes `adaptive` from 0860's unavailable-provider validation and adds an `else if (provider == "adaptive")` init-selection branch calling `ggml_backend_cuda_comm_init_hybrid(ret)`.
- Assert application/idempotence and missing-anchor fail-closed behavior; assert the resulting selector path contains no `GGML_CUDA_ALLREDUCE=hybrid` dependency.

## Hardware validation
Hardware/workload: dual gfx1100 (2x Radeon RX 7900 XTX), Qwen3.8-27B-Q8_0, `-sm tensor`, RCCL available. Enable `BIGCHERRY_PATCH_TRACE=1`; `--allreduce adaptive` must start and emit the 0860 activation marker resolving provider `adaptive` with native wire, with hybrid initialization/runtime evidence captured.

Run order-balanced A/B comparisons for `adaptive` vs `ccl` and `adaptive` vs `host`, holding build/model/workload/settings constant. Require greedy-output parity and MTP draft-acceptance parity before interpreting performance.

Exercise reductions on both sides of the adaptive crossover. Small messages below the internal pipeline copy threshold should prefer the internal/host path; large messages at/above the threshold should prefer RCCL/ccl when available. Verify effective-provider telemetry agrees with message size and that provider failure follows the documented fallback chain rather than silently changing the requested CLI selection.

Report pair order, reduction-byte ranges/effective provider, acceptance counts/ratio, throughput deltas, and statistical result. Treat any correctness or activation mismatch as a failed/void performance comparison.
