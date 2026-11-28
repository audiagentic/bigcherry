---
id: ENG04
order: 0
plan: tuning-radiance-engine
state: pending
created-at: '2026-11-10T13:12:01.542345+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P3
work: L
---

# KV format and context-capacity matrix on the current build: fp8 / fp8_v4 / k6_v4, 262K concurrency, single-session 1M

## Description

The host pool/KV tier reallocation (RSC03, #512) is delivered and the 1M-context tail-size matrix is recorded (RHP10), but no matrix exists on the current pin for the KV formats (fp8 default, fp8_v4 0.78x, k6_v4 0.69x per token: both lossy), the full 262K concurrent-session limit, retained-session behaviour after long residency, or whether a single 1M-token session fits without costing ordinary multi-stream throughput. Aggregate 1M-token fills across eight sessions are not a 1M single session.

## Steps

1. Per format: capacity (tokens), prefill and decode at 52K/370K/729K used context, quality (KL or long-context accuracy), host-memory admission.
2. 262K x 8 sessions: concurrency and decode step.
3. Single 1M session with yarn:4: fit, prefill, decode.
4. Per-arm short lease windows.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A table per format and context depth with the policy state, plan key and build recorded; quality for the lossy formats against the BF16 reference.

## Effort & Risk



## Standards



## Acceptance Criteria

Measured capacity, throughput and quality per KV format on the current build, and a recorded decision on whether any lossy format should be offered.

## Notes

Source: gateway prior-work audit 2026-11-10 (missing item 5). Old llama.cpp memory ceilings (QFN03/04) must not be copied into Radiance limits.

RV4754: qualify existing approved FP8/BF16/BF16-I8 formats and capacity boundaries in affected cohorts; new formats and comprehensive 1M/context matrix after baseline. Shared mode remains RSC16-18.

## Change Log

- 2026-11-10T13:12:01.542345+00:00 (created-by): Created by agent
- 2026-11-27T15:13:58.181503+00:00 (updated-by): Updated: section:notes

## Notes Append

2026-11-28: narrowed to the cache-format quality and speed decisions (fp8, fp8_v4, k6_v4 against bf16/bf16_i8), retained-session behaviour and supported context limits; capacity and speed numbers come from ENG09's 2.3 baseline instead of a separate matrix. A 1M single session is not a qualified capability of the 262,144-context Qwen checkpoint. Shared capacity stays RSC16-18; XDP02 stays separate (it changes which device owns attention layers).
- 2026-11-28T12:55:37.692832+00:00 (updated-by): Updated: section:notes_append
