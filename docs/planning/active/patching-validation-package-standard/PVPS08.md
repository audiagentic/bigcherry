---
id: PVPS08
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-25T23:16:40.340989+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Timed-process PCI attestation via hipDeviceGetPCIBusId logging

## Description

GPT review: the -sm layer preflight proves physical-device membership but not that the timed tensor process resolved ordinals identically. Add a small framework patch that logs each HIP backend device's PCI bus id (hipDeviceGetPCIBusId) at ggml-cuda init, so every timed process attests itself; then drop the preflight.

## Steps

1. Framework patch: GGML_LOG_INFO 'ggml_cuda_init: device N pci=<bdf>' per device.
2. parse_llama_server_attestation reads it (locator per index).
3. Remove tensor_split_preflights once all campaign builds carry it.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Unit test on a synthetic log; one dual-XTX session attests without preflight.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-25T23:16:40.340989+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260925_232233_correctness-checks-for-approxi_1217
- 2026-09-25T23:22:42.336192+00:00 (updated-by): Updated: section:ledger-events
