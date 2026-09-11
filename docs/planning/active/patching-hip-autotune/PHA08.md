---
id: PHA08
order: 8
plan: patching-hip-autotune
state: pending
created-at: '2026-09-11T22:57:38.584940+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# HIP Flash-Attention D=72 VLM aperture-violation workaround (#28664)

## Description

Evaluate the reported 2x RX 7900 XTX HIP tiled Flash-Attention D=72 aperture violation for vision workloads and qualify a narrow fallback only if the current pin reproduces it.

## Steps

1. Resolve PR #28664 status, head, ancestry, and exact affected source. 2. Reproduce on supported XTX/VLM image sizes with native controls and capture HSA fault evidence. 3. If absent upstream, author a package-only narrow patch that disables HIP FA only for D=72 under the proven predicate and falls back to matmul attention. 4. Validate correctness, no HSA faults, and unchanged non-D=72/text-only behavior. 5. Keep default enablement and promotion gated until complete evidence exists.

## Detailed Solution & Technical Design

This is a patch qualification item, not a generic FA disable. The reported issue is specific to HIP tiled FA, D=72, larger VLM images, and 2x XTX; post-workaround success is external evidence only. Reuse package-only patch contracts, current-pin ancestry, fail-closed apply/idempotence, correctness evidence, and promotion gates. Retire any local patch if an equivalent upstream fix is pinned.

## Code Samples & Guidance



## Files

patches/<new-id>/**; tools/bigcherry/patch/**; tools/bigcherry/tuning/correctness_evidence.py; tools/tests/patch/**; tools/tests/tuning/**; docs/evidence/<run-id>/

## Validation

PR/current-pin ancestry; 1024/2048/2560px image matrix on 2x XTX; D=72 fault absence; numerical/output parity against matmul fallback; non-D=72 and text-only controls; patch apply/idempotence/rebase; promotion evidence.

## Effort & Risk



## Standards



## Acceptance Criteria

Either current-pin/non-reproduction evidence closes the gap without a patch, or a narrow package-only workaround passes apply/idempotence, VLM correctness, fault absence, and non-regression gates. No broad HIP FA disablement is introduced.

## Notes

Provenance: shared ChatGPT conversation, 11 Sep 2026, '#28664 — direct 2×7900 XTX HIP Flash-Attention crash'; source https://github.com/ggml-org/llama.cpp/pull/28664. Existing PHA06 is a different completed item; this is a new D=72 issue.

## Change Log

- 2026-09-11T22:57:38.584940+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260911_225756_added-six-provenance-rich-buil_2622
- 2026-09-11T22:57:56.719160+00:00 (updated-by): Updated: section:ledger-events
