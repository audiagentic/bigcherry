# Testing — PRBE41 SSM_CONV channels-major (1263)

## Applicability (authoritative 2026-10-11)

- **Positive lane:** single physical gfx1100 or gfx1201 GPU, HIP, Qwen hybrid GDN/SSM, `model.split_mode() != LLAMA_SPLIT_MODE_TENSOR`. Channels-major input is enabled by the existing 1263 graph selector.
- **Negative lane:** `-sm tensor` on dual XTX or mixed-RDNA. `1263/patch.py::_TIME_MAJOR_FALLBACK` deliberately restores stock time-major graph and emits `ssm_conv_channels_major_declined_split`. The positive marker **must not** be interpreted as expected in this lane. PRBE114 owns any later Meta split-state work; do not change it during QFP49 activity.
- **Unsupported backend:** 1263 does not port the AMD Metal `supports_op` rejection. Do not run or claim Metal correctness until that guard is qualified. Vulkan/SYCL/OpenCL/CANN/Hexagon/WebGPU have explicit channels-major rejection in the patch.
- **State:** conv-state bytes are layout-specific; never resume, restore or reuse subject state in control, or vice versa.

## Hardware-free gates (not executed in the 2026-10-11 audit)

`PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --focal-overlay 1263_prbe41_ssm_conv_channels_major`, patch lint, direct `test-backend-ops -o SSM_CONV`, producer contract validation. Check d_conv 3/4/9, d_inner multiples of 128, token lengths 1/2/7/33 and long prefill, n_seqs 1/2/4, time-major vs channels-major reference, recurrent K-slot updates and graph replay.

Audit-only static checks 10/10 and host layout/state-tail fixtures 144/144 passed. These are **not** C++/HIP execution.

## Activation and correctness

`BIGCHERRY_PATCH_TRACE=1`: on the single-device subject require `BIGCHERRY_PATCH_HIT patch=1263_prbe41 path=ssm_conv_channels_major` and its absence in the control. Under tensor split require the `...path=ssm_conv_channels_major_declined_split` marker and no actual channels-major kernel execution. A once-per-process log marker does not prove the benchmark lane itself activated; correlate request/graph/physical device and profiler kernels.

Reuse `validation/producer.py` full-vocabulary backend_reference; verify same-model logprobs/KLD, greedy tokens, multi-request same-process, multi-ubatch, long context, MTP n_max 4 acceptance, accepted-token state publication, snapshot rollback, capture/replay and memory safety. Fail closed if work or copies unexpectedly disappear.

## Existing measured evidence and next cost gate

`evidence/validation.json` has 24 historical b11126/b11233 records; `evidence/withdrawn.json` excludes invalid/parallel runs. Four validated gfx1100 b11233 pp512 sessions measured +1.1268% to +1.4341% geometric throughput (CI95-low +0.8969%..+1.1974%). Maximum historical gfx1201 CI95-low across records: +2.5859%. These do not qualify b11474 or satisfy >=3% promotion.

Before scheduling new hardware, inspect existing queues and capture actual current-pin exposed CONCAT/CONT/SSM_CONV time and bytes. If the removable time is <2.9126% of E2E, **REJECT** further implementation. Only if a larger ubatch survives: pp512/1024/4096 with explicit `-b/-ub`, fixed MTP and context, tg128 non-SSM control, four independent sessions/architecture, >=10 paired ABBA rounds/session. Promote only CI95-low >=3% E2E, <=1% control regression, full numerical and state parity. Do not overlap active QFP49 experiments.

## Ownership and external reference

PRBE41/1263 owns the single-device layout decision; PRBE114 owns Meta tensor-split propagation; QFP49/1360/1361 owns active cross-card/stagger scheduling. External [AMD #52](https://github.com/AMD-Ecosystem/llama.cpp/pull/52) shows gfx1151 results, not transferable AMD fleet gains. Open [upstream #30149](https://github.com/ggml-org/llama.cpp/pull/30149) is Vulkan transposed-CONCAT tiling, not a HIP replacement.
