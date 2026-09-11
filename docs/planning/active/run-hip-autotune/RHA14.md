---
id: RHA14
order: 2
plan: run-hip-autotune
state: pending
created-at: '2026-09-11T22:57:06.182053+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# HIP staggered multi-sequence correctness regression gate (#28537)

## Description

Preserve and formalize the reported HIP -np >1 staggered-request corruption case, which was closed upstream without a linked fix and therefore must not be treated as resolved.

## Steps

1. Resolve the issue and reconstruct the staggered second-sequence prompt/decode reproducer. 2. Run HIP -np >1 with staggered requests, plus single-sequence HIP, CPU, and Vulkan controls. 3. Compare deterministic outputs/token hashes and retain the failure receipt. 4. Add the check to server validation before trusting concurrent-server performance evidence.

## Detailed Solution & Technical Design

Reuse existing server-runner correctness and evidence contracts. This item owns the regression gate and reproducible workload, not an assumed fix. A closed issue without a linked fix is an unresolved external status, not implementation evidence.

## Code Samples & Guidance



## Files

tools/bigcherry/tuning/server_runner.py; tools/bigcherry/tuning/correctness_evidence.py; tools/bigcherry/campaign/benchmark.py; tools/tests/tuning/**; tools/tests/campaign/**; docs/evidence/<run-id>/

## Validation

Issue #28537 status/linked-PR check; deterministic staggered-request matrix; output/token-hash parity; control separation; current-pin and runtime provenance; no benchmark admission when correctness fails.

## Effort & Risk



## Standards



## Acceptance Criteria

The HIP -np >1 staggered-request behavior has a committed pass/fail evidence record and is enforced as a prerequisite for concurrent-server performance claims. Closure of the upstream issue alone cannot clear the gate.

## Notes

Provenance: shared ChatGPT conversation, 10–12 Sep 2026, '#28537 — status changed, but don't treat it as fixed'; source https://github.com/ggml-org/llama.cpp/issues/28537. Report states corruption remains reproducible and no linked fix exists. No matching current BigCherry plan found by scan.

## Change Log

- 2026-09-11T22:57:06.182053+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260911_225756_added-six-provenance-rich-buil_2622
- 2026-09-11T22:57:56.704697+00:00 (updated-by): Updated: section:ledger-events
