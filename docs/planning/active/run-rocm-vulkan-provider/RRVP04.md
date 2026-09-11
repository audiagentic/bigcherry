---
id: RRVP04
order: 4
plan: run-rocm-vulkan-provider
state: pending
created-at: '2026-09-11T22:57:41.597815+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Vulkan async-copy synchronization reduction A/B (#28618)

## Description

When Vulkan scope resumes, qualify merged upstream PR #28618's reduction of CPU↔GPU/full-fence synchronization for small per-token input transfers, especially across no-P2P split boundaries.

## Steps

1. Resolve merge ancestry and choose parent-of-merge versus merge/current-master boundary. 2. Run single-R9700 control, 2x XTX, and mixed/no-P2P split cases with tg128/tg512 across Qwen3.6 27B and 35B-A3B. 3. Capture throughput plus inter-split copy/synchronization time and GPU utilization. 4. Require correctness and exact build/runtime/device provenance; compare gains against single-GPU controls. 5. Do not implement a local patch if the change is already in the selected upstream baseline.

## Detailed Solution & Technical Design

This is a Vulkan run qualification item and remains paused with the rest of the Vulkan provider chain. The shared report says #28618 merged 10 Sep and changes idle input downloads from GPU async copy plus fence to CPU writes, with submitted RTX results establishing direction rather than AMD proof. Reuse RRVP02 identity, RRVP03 attestation, existing campaign A/B, and no-P2P topology evidence; do not create a parallel synchronization policy.

## Code Samples & Guidance



## Files

tools/bigcherry/campaign/**; tools/bigcherry/profiling/**; tools/bigcherry/tuning/correctness_evidence.py; tools/tests/campaign/**; docs/evidence/<run-id>/

## Validation

Upstream ancestry; parent/merge exact build identity; single and multi-GPU Vulkan controls; copy/fence telemetry; output parity; no-P2P split topology; no local duplicate when pinned upstream already contains the fix.

## Effort & Risk



## Standards



## Acceptance Criteria

A committed Vulkan evidence package establishes the AMD/no-P2P impact of #28618 or records a justified no-effect result. Correctness, synchronization telemetry, and exact source/build/runtime identity are present; no duplicate local implementation is added.

## Notes

Provenance: shared ChatGPT conversation, 11 Sep 2026, '#28618 — merged Vulkan transfer/synchronization fix'; source https://github.com/ggml-org/llama.cpp/pull/28618. Implementation is paused until the existing Vulkan provider identity chain resumes.

## Change Log

- 2026-09-11T22:57:41.597815+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260911_225756_added-six-provenance-rich-buil_2622
- 2026-09-11T22:57:56.723364+00:00 (updated-by): Updated: section:ledger-events
