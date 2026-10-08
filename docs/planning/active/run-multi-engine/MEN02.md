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

# Radiance basics working: serve Qwen3.8-27B on the R9700 and measure it our way

## Description

First working lane on the second engine, kept to what can run today: the source build from MEN01 serving the MXFP4 Qwen3.8-27B checkpoint on the R9700, measured with the lab client and corpus. Then the limits: Flash-Next, longer context, the other cards.

## Steps

1. Launch the MEN01 build with /models/Qwen3.8-27B-MXFP4-mtpfp8 on the R9700; record the full command and settings.
2. Measure 2K / 8K / 24K / 98K with tools/lab/reference-vllm/bench-openai.py, two repeats, with and without the DFlash2-FP8 drafter; compare against the container figures and the llama.cpp single-card figures.
3. Check the llama-server-native API: does tools/lab/flash-next/long-ctx-profile.sh's request path (/completion, timings, cache_prompt) work unchanged against radiance? List what differs.
4. Limits: largest context that loads; whether a Flash-Next checkpoint exists or can be produced in a supported format; what happens on an RX 7900 XTX.
5. Hand prompts and outputs to MEN05 for the quality comparison.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per configuration: loads or the exact failure, context reached, VRAM, prefill and decode t/s, acceptance.

## Effort & Risk



## Standards



## Acceptance Criteria

A radiance lane built from source reproduces the container's prefill within run-to-run spread on the R9700, with the list of what does not yet work (models, cards, API differences).

## Notes

KV precision rule applies: fp8 is acceptable, nothing below 8 bits. Depends on MEN01 only for knowing which settings are safe to vary.

## Change Log

- 2026-10-08T20:36:07.241630+00:00 (created-by): Created by agent
- 2026-10-08T20:47:49.090866+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:validation, section:acceptance_criteria
