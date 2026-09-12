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

Implement and enforce the HIP staggered multi-sequence correctness regression gate through ServerRunner, preserving the unresolved external issue as a regression guard until a reproducible pass is recorded.

## Steps

1. Freeze the workload contract: request count, stagger schedule, prompts/seeds, model/build/pin, HIP device visibility and expected output/token-hash comparison.
2. Reconstruct the staggered second-sequence prompt/decode reproducer.
3. Run HIP -np>1 staggered requests plus single-sequence HIP and CPU controls; Vulkan is a separate comparative boundary and not a dependency.
4. Persist deterministic outputs/token hashes, failure classes, runtime/build provenance and terminal receipts.
5. Add the gate to concurrent-server validation before performance evidence can be admitted; use ServerRunner, never a standalone script.

## Detailed Solution & Technical Design

Reuse ServerRunner, correctness_evidence.py and campaign benchmark/evidence contracts. Define a versioned staggered workload and failure taxonomy (mismatch, timeout, process failure, incomplete receipt, environment mismatch). A closed upstream issue without linked fix is not resolution; preserve a failing receipt and block performance admission.

## Code Samples & Guidance



## Files

tools/bigcherry/tuning/server_runner.py; tools/bigcherry/tuning/correctness_evidence.py; tools/bigcherry/campaign/benchmark.py; tools/tests/tuning/**; tools/tests/campaign/**; docs/evidence/<run-id>/

## Validation

Issue #28537 state and exact reproducer; deterministic staggered matrix; output/token-hash parity; single/CPU controls; current-pin/runtime provenance; no concurrent benchmark admission on correctness failure.

## Effort & Risk



## Standards

ServerRunner reuse; deterministic workload; explicit failure classes; no standalone driver; Vulkan controls remain separate.

## Acceptance Criteria

A committed reproducible pass/fail evidence record exists, concurrent-server performance is gated on it, and upstream issue closure alone cannot clear the gate.

## Notes

Provenance: shared ChatGPT conversation, 10–12 Sep 2026, '#28537 — status changed, but don't treat it as fixed'; source https://github.com/ggml-org/llama.cpp/issues/28537. Report states corruption remains reproducible and no linked fix exists. No matching current BigCherry plan found by scan.

Provenance: upstream issue #28537 and prior shared review. GPT says no canonical staggered driver was found; contract must be frozen before implementation.

## Change Log

- 2026-09-11T22:57:06.182053+00:00 (created-by): Created by agent

## Ledger-events


- chg_20260911_225756_added-six-provenance-rich-buil_2622
- 2026-09-11T22:57:56.704697+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T10:31:45.140927+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260912_103200_updated-the-active-buildrunp_8222
- 2026-09-12T10:32:01.042667+00:00 (updated-by): Updated: section:ledger-events
