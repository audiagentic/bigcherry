---
id: PWR13
order: 0
plan: run-power-efficiency
state: pending
created-at: '2026-11-26T01:45:56.843677+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Expand independent GPU qualification workloads and fail-closed proof

## Description

Extend existing platform gpu-ai-screen beyond repeated BLAS and sequential memory. Independent coverage only; no engine dependency or duplicate controller/profile/store. PWR12 retains physical research; RCS26–28 own native replay; PWR07/08/RSC02 own admission/recovery; PWR10/PWR02 own production.

## Steps

1. Review owner admission/restore before unattended mutation. 2. Typed cases, diverse CPU oracles and verifier fault checks. 3. Capability-proven instruction, memory, transfer, mixed and transition cases. 4. Managed evidence; observe-only hardware cohort then approved finalist tests.

## Detailed Solution & Technical Design

docs/design/gpu-qualification-code-guide.md sections 1–12.

## Code Samples & Guidance



## Files

tools/bigcherry/hardware/ai_probe.cpp, ai_probe.py, cli/ai_probe_cmds.py, tools/tests/hardware/test_ai_probe.py; existing power owners by coordination.

## Validation

CPU vectors, fault injection, malformed coverage, target compile/ISA proof, leased observe-only cases on supported cards, exact restore and matched finalist receipts.

## Effort & Risk



## Standards



## Acceptance Criteria

Required cases complete with positive verified work; missing/unsupported/thermal/recovery nonpass. No unattended writes until recovery gates pass. No auto promotion; native and serving gates separate.

## Notes

## Change Log

- 2026-11-26T01:45:56.843677+00:00 (created-by): Created by agent
