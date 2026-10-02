---
id: MET05
order: 5
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-02T04:45:07.116948+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# 1284 aux_rocm_expert_backend: 6900 XT (ROCm3) as auxiliary expert device outside Meta/RCCL

## Description

Let ROCm3 hold and compute cold experts without joining the tensor-split Meta buffer type or RCCL/AllReduce: scheduler auxiliary device list (--expert-aux-device ROCm3), simple ROCm buffers, explicit pinned-host staging primary->host->GPU3 and GPU3->host->GPU0 (~10 KiB each way per layer), GPU0 aggregation. Later test plain HIP peer copies separately (no PCIe atomics does not imply DMA peer copies fail).

## Steps

0. Diagnose the earlier -ts 3,3,2,0 -ot exps=ROCm3 failure: a broad exps regex moves ALL routed experts (60+ GiB IQ4_XS) onto the 16 GB 6900, so OOM is the first hypothesis, ahead of RCCL. Re-test with one layer only: -ot '^blk\.0\.ffn_(gate|up|down)_exps\.weight$=ROCm3' under -sm layer then -sm tensor, logging selected buft and first allocation failure.
1. Aux backend registration after Meta backend creation.
2. Host-staged copies; latency instrumentation.
3. Placement tier kind aux_device.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

ROCm3 not in RCCL communicator (trace); greedy parity + KLD; decode with 6900 tier vs CPU tier for the same cold experts; copy latency per layer.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

The 6900's value is absorbing many individually cold experts so CPU route mass stays <= 0.25-0.5%.

## Change Log

- 2026-10-02T04:45:07.116948+00:00 (created-by): Created by agent
