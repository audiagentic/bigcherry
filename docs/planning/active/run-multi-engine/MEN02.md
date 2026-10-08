---
id: MEN02
order: 2
plan: run-multi-engine
state: pending
created-at: '2026-10-08T20:36:07.241630+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Feasibility: the models we serve on vLLM on this box (context, cards, quality)

## Description

Second gate. Every Flash-Next number we have is llama.cpp (245,760 context over two RX 7900 XTX + R9700, drafter on the RX 6900 XT). Nothing shows that vLLM can serve Flash-Next here, at what context, or on which cards. The 27B result is one R9700 only.

## Steps

1. Checkpoint availability for Qwen3.8-Flash-Next in a format the stack loads (MXFP4 / FP8); record source and size. A DSpark drafter was already converted with the pinned converter on 2026-10-08.
2. Largest context that loads on the R9700 alone, and with --pipeline-parallel-size over the mixed cards; whether --tensor-parallel-size 2 runs on the two XTXs with this stack at all.
3. Speed at 2K / 8K / 24K / 98K with tools/lab/reference-vllm/bench-openai.py, same corpus as the llama.cpp runs.
4. Hand the outputs to MEN05 for the quality comparison.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per configuration: loads or the exact failure, context reached, VRAM per card, prefill and decode t/s, acceptance.

## Effort & Risk



## Standards



## Acceptance Criteria

A yes/no per model and card set, with numbers, sufficient for the owner to decide whether vLLM becomes a production lane.

## Notes

KV precision rule applies: fp8 is acceptable, nothing below 8 bits. Depends on MEN01 only for knowing which settings are safe to vary.

## Change Log

- 2026-10-08T20:36:07.241630+00:00 (created-by): Created by agent
