---
id: MEN05
order: 5
plan: run-multi-engine
state: pending
created-at: '2026-10-08T20:36:28.043533+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Cross-engine quality check: same prompts, comparable distributions

## Description

The speed comparison is meaningless without a quality one: the vLLM lane runs 4-bit MXFP4 weights with fp8 KV, ours Q8_0 or IQ4_XS with f16 KV. flash-fidelity.sh compares llama.cpp builds against a CPU f32 reference through llama-server internals; nothing compares across engines.

## Steps

1. Probe set through the OpenAI API only: fixed prompts at several depths, logprobs of the top-k next tokens, greedy continuation text.
2. Reference = llama.cpp CPU f32 (or BF16) on the same prompts; report top-1 agreement and total-variation distance per engine against it, as flash-fidelity.sh does today.
3. Task-level check: a small fixed set of long-context questions with known answers, scored identically for both engines.
4. Record the result beside every cross-engine speed figure.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

tools/lab/flash-next/flash-fidelity.sh, tools/lab/reference-vllm/

## Validation

The check reproduces the existing llama.cpp fidelity numbers when both arms are llama.cpp builds (production 23/24 top-1, TV mean 0.076 at 8K), then runs against the vLLM container.

## Effort & Risk



## Standards



## Acceptance Criteria

Every cross-engine speed comparison in releases/evidence has a quality line produced by this check.

## Notes

Independent of MEN03: can start as a plain lab script using the two existing launchers.

## Change Log

- 2026-10-08T20:36:28.043533+00:00 (created-by): Created by agent
