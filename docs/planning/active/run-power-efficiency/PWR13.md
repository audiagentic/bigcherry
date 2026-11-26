---
id: PWR13
order: 0
plan: run-power-efficiency
state: in_progress
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

2026-11-26 re-review and first implemented slice: rotating dense/sparse/signed-diagonal exact corpus, guarded output and per-corpus host/device verification, input readback, comparator faults at three positions, native protocol 2 / observation schema 3; old reference and stale binary refused before candidate transaction. ROCm 7.2.4 builds gfx1030/gfx1100/gfx1201, CPU self-test, 197 hardware tests pass (one skipped), 774373 clean GEMMs over four physical cards and 28 full 128 MiB memory passes. Six observe-only managed captures harvested: 20261126T015614Z-gpu-ai-screen-512f35e587ee, 20261126T015751Z-gpu-ai-screen-82c5b9fae077, 20261126T015805Z-gpu-ai-screen-5cb973591371, 20261126T015818Z-gpu-ai-screen-835094a5e951, 20261126T015836Z-gpu-ai-screen-6c7afecb7598, 20261126T015921Z-gpu-ai-screen-4880e38ee7de. Detailed counts in AI_POWER_SCREEN.md. PWR13 stays in progress: varied shapes, direct instruction/wave/LDS, mixed traffic and hot/transient scopes outstanding; P0 owner admission and exact restoration fixes required before unattended mutation. No control or native owner edits.

## Change Log

- 2026-11-26T01:45:56.843677+00:00 (created-by): Created by agent
- 2026-11-26T01:52:35.520482+00:00 (state-transition): State: pending → in_progress
- 2026-11-26T02:00:20.973947+00:00 (updated-by): Updated: section:notes
