---
id: BCOP112
order: 112
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-10T09:09:00+00:00'
created-by: agent
priority: P3
work: S
---

# PRBE21: gate CONCAT fusion on recurrent-state publication

## Discovery / disposition

Pinned b11474 `build_conv_state` materializes `conv_input` for SSM_CONV **and** recurrent-state VIEW/CPY. A conv-only CONCAT-elimination leaves state updates unwritten; existing CUDA fusion cannot skip a non-adjacent CPY. Retain native producer and stock SSM_CONV+SiLU. Do not allocate the proposed `12xx` patch or add a new dispatch/config surface. No measured gain.

## Authoritative owners / exclusions

PRBE21 owns CONCAT+CPY state producer; PRBE18 owns downstream pre-scan; PRBE41/1263 owns channels-major SSM layout. Small-K MMVQ rpb=2 remains a separate PRBE21 candidate, unchanged. Last independent PRBE21/PRBE18 plan update 2026-10-08 06:47 UTC; no independent implementation/PR/queued experiment found in the preceding 12 hours. Protected Radiance gfx1100 MXFP4/serve, Flash-Next 1330/1334/1347, router 1357, QFP35/41/43 and MTP work untouched. Earlier BCOP111 and prior audit slices not repeated.

## Next gate / terminal outcome

Use existing PRBE113 profiling when available; no duplicate queue. Prove actual CONCAT+CPY graph adjacency, state lifetime and non-overlapped E2E share >=2.9126%. Otherwise **close the fold without code**. A surviving default-off producer+CPY fusion must preserve exact state/graph/replay, full logits/greedy/MTP, and >=3% CI95-low E2E with <=1% controls across four sessions/architecture. No separate cache/allocator/scheduler.

## Evidence

Pinned b11474 `delta-net-base.cpp`, `qwen35.cpp`, `concat.cu`, `ssm-conv.cu`, `ggml-cuda.cu`; upstream #30198, vLLM causal_conv1d, SGLang #38623. 12/12 source assertions and 35+35 disposable host-model cases passed. No HIP compile, GPU tests or hardware measurements. Technical algorithm and controls live in PRBE21.
