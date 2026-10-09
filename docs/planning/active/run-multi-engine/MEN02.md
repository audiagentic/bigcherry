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

2026-10-09 first working lane (steps 1 and 2 done). Source build of radiance 1.3.0 (89cee7ce; ROCm 7.2.4, g++ 14.2 unpacked privately) serving the published container StillDeadcode/qwen3.8-27b-mxfp4 (19,377,119,232 bytes, sha256 a4908940...df4f8 verified) on the R9700 alone: --tp 1 --max-num-seqs 8 --max-model-len 180000 --kv-cache-dtype fp8. Measured with tools/lab/reference-vllm/bench-openai.py, two repeats (script tools/lab/radiance/run-radiance.sh). Prefill t/s at ~2.3K / ~9K / ~35K / ~111K prompt tokens: 3,088-3,267 / 2,565-3,158 / 2,801-2,834 / 2,050-2,069 (same with the drafter off). Decode t/s with the container's DFlash2 drafter (auto): 106-134 / 96-99 / 96-106 / 77-80, acceptance 34.7% overall (position 0: 77.9%, position 1: 74.3%). Decode with the drafter off: 37.9 / 37.1 / 34.4 / 28.5. VRAM in use on the R9700 about 34.1 GB (the card's full reported capacity). Against BigCherry llama.cpp on the same card (UD-Q4_K_M, f16 KV): prefill 2.4x ours (1,141-1,291 to 35K, 844-857 at 111K, no drafter); decode with no drafter 20-25% faster (30.5 / 28.1 / 23.0); decode with a drafter about 2x our MTP (51-69 / 49-52 / 32-44). Against the vLLM + radiance container measured earlier: same prefill to 35K (~3,000), lower at 111K (2,050 against ~2,470), higher decode at 111K (77-80 against ~50). Operational findings: radiance's built-in Hugging Face fetch was cancelled mid-stream by the server and restarted from zero, and its helper (libexec/radiance/hfget.sh + curl) survives a kill of the server and keeps the queue's lock file descriptors open; fetch containers outside the GPU lock with a resumable client and verify size and sha256 before serving. Remaining steps: 3 (llama-server-native API against long-ctx-profile.sh), 4 (limits: Flash-Next container needs two RDNA4 cards for --tp 2; --tp 1 with expert tiering untested; an RX 7900 XTX has no kernel library), 5 (quality, MEN05).

## Change Log

- 2026-10-08T20:36:07.241630+00:00 (created-by): Created by agent
- 2026-10-08T20:47:49.090866+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:validation, section:acceptance_criteria
- 2026-10-09T01:50:33.540603+00:00 (updated-by): Updated: section:notes
