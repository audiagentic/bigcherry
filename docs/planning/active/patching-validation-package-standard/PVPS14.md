---
id: PVPS14
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-28T22:28:32.391094+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# 1005 prompt-cache-checkpoint-selection: unresolved correctness failure + producer activation-evidence gap

## Description

GPT design review (reviewer-gpt-agent, req_0b8ef25771864ee1, 2026-09-28) of patch 1005_prompt_cache_checkpoint_selection's contract + producer, requested after a real gfx1100 hardware run found a genuine correctness divergence (63/64 tokens matched, diverged only at the final token). Disposition: REQUEST CHANGES -- state stays untested. Contract design itself (greedy_parity primary gate, state_restore workload tag reuse, ci95_threshold_bound_v1 for the auxiliary decode lane only) is sound.

## Steps

1. Producer protocol gap (primary blocker): producer.py primes checkpoint A then assumes cold-reference run C exercised the shorter-prefix checkpoint-reuse path, but never proves it -- trace_probe="skip" and promotion_trigger_evidence is hardcoded positive_hit=False. A PASS could occur even if no checkpoint was actually reused. Need real observable evidence (server log/API) that C found/reused a prompt-cache checkpoint with a shorter common-prefix position than A's stored checkpoint -- add patch-local diagnostic instrumentation if no stable existing log line encodes this.
2. The 63/64-token divergence is a genuine unresolved correctness failure, not yet distinguishable from FP-tie-breaking noise. GPT's required discriminating evidence: capture cold vs warm full-vocabulary logits (or at minimum top-2 token IDs/logits) at the first divergent prediction, across several repeated fresh warm/cold runs. Stable logit displacement across repeats => real state/cache defect. Tie-flip under near-identical logits => FP sensitivity, more benign. Do NOT weaken greedy_parity or add a token-mismatch tolerance before this evidence exists.
3. Doctrine-wording inconsistency: patch/contract comments describe 1005 as "correctness-only" but validation.toml requires performance+controls and the producer runs paired llama-bench + emits a promotion lane. Either reword to "correctness-gated with auxiliary no-regression evidence" or remove the performance claim -- do not let it silently participate in disposition while being described as absent.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Dispatched dev-gpt-agent (req_0356c1e3b9f74d2a) to implement fixes for the review's two mechanical gaps. It delivered real code (branch fix/pvps14-pccs-evidence, merged b821fd68): server-log proof of shorter-prefix checkpoint reuse, and conditional top-2 token/logprob divergence capture across the original pair + 3 fresh warm/cold repeats. Found and fixed two real bugs in the draft before merging (GPT had no execution environment to catch these): activation_evidence was a plain dict, not the required ActivationEvidence instance (would crash); the checkpoint-detection regex targets SLT_TRC/SRV_TRC server logs which are trace-level (verified in vendor/llama.cpp's common/log.h) and don't print at the default verbosity threshold (would silently always fail-closed) -- added -v to the warm server args (verified against common/arg.cpp). Offline suite clean. State stays untested -- the actual 63/64-token divergence question is still open pending a real hardware run with this new diagnostic capture.

## Change Log

- 2026-09-28T22:28:32.391094+00:00 (created-by): Created by agent
- 2026-09-28T22:28:44.022196+00:00 (updated-by): Updated: section:description, section:steps

## Ledger-events

- chg_20260928_224951_fixed-1005s-validation-produc_9562
- 2026-09-28T22:49:54.050331+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-28T22:50:02.035924+00:00 (updated-by): Updated: section:notes
