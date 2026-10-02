---
id: RRVP04
order: 4
plan: run-rocm-vulkan-provider
state: superseded
created-at: '2026-09-11T22:57:41.597815+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Superseded: Vulkan async-copy #28618 isolated A/B

## Description

c061 already contains the idle-backend CPU-copy optimisation. Its telemetry folds into RRVP05 rather than a separate parent-vs-merge campaign.

## Steps

1. Record c061 source evidence that the optimisation is present.
2. Move copy/synchronisation telemetry requirements into RRVP05.
3. Keep a historical pointer to #28618 only for regression attribution.

## Detailed Solution & Technical Design

This is a Vulkan run qualification item and remains paused with the rest of the Vulkan provider chain. The shared report says #28618 merged 10 Sep and changes idle input downloads from GPU async copy plus fence to CPU writes, with submitted RTX results establishing direction rather than AMD proof. Reuse RRVP02 identity, RRVP03 attestation, existing campaign A/B, and no-P2P topology evidence; do not create a parallel synchronization policy.

## Code Samples & Guidance



## Files

tools/bigcherry/campaign/**; tools/bigcherry/profiling/**; tools/bigcherry/tuning/correctness_evidence.py; tools/tests/campaign/**; docs/evidence/<run-id>/

## Validation

No local duplicate patch; RRVP05 captures current pinned behaviour.

## Effort & Risk

S / low.

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
- 2026-10-02T12:34:50.580137+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:validation, section:effort_risk
- 2026-10-02T12:35:16.360298+00:00 (state-transition): State: pending → superseded
