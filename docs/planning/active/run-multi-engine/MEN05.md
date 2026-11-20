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

Cross-engine speed comparisons require quality evidence tied to exact engine/source, model checkpoint and pack semantics, tokenizer, quant, KV, context, MTP and sampling. Logit/top-k metrics apply only where outputs are numerically comparable; unmatched quant/model arms require task-quality reporting and must not be presented as a quant-neutral engine comparison.

## Steps

1. Record source/build/ABI, model/checkpoint and pack digests, tokenizer and prompt IDs, quant/KV/context, effective feature state and MTP settings for every arm. Label MATCHED or UNMATCHED.
2. For comparable logits, measure finite outputs, next-token/top-1 agreement and TV/KLD against an explicit reference build/model at several context depths. Keep existing llama.cpp fidelity reproduction as a separate regression check.
3. Run fixed greedy and sampled continuations with temperature/top-k/top-p and RNG policy recorded. Include cold/warm and multi-turn cases where relevant.
4. For MTP, record requested and active depth, proposed and accepted tokens per step/distribution, verification cost and effective target tokens/s; requested MTP is not evidence it was active.
5. Compare a fixed set of realistic coding-agent tasks with blinded or rubric-based correctness, solve/refusal/error rate and end-to-end time-to-solution. For unmatched quant/model arms report a quality/performance tradeoff, not an engine speed ratio.
6. Attach the quality record to every paired speed result; report repetitions and dispersion.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

tools/lab/flash-next/flash-fidelity.sh, tools/lab/reference-vllm/

## Validation

Existing llama.cpp fidelity reproduction reports model/build, top-1 and TV/KLD. New records include exact arm identity, comparability label, sampling/RNG, MTP activation and accepted-token distribution, and coding-task scoring/time-to-solution. Invalid/nonfinite output fails the run.

## Effort & Risk



## Standards



## Acceptance Criteria

Every cross-engine performance comparison is accompanied by quality data bound to immutable arm identity. Comparable logits include top-1 and TV/KLD. Sampled generation, MTP behavior and realistic coding-task outcomes/time-to-solution are measured where applicable. Unmatched arms are reported explicitly without quant-neutral speed ratios.

## Notes

Independent of MEN03: can start as a plain lab script using the two existing launchers.

## Change Log

- 2026-10-08T20:36:28.043533+00:00 (created-by): Created by agent

## Reviews

- RV4236
- 2026-11-20T02:20:43.995620+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
