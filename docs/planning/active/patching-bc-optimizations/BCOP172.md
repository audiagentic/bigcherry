---
id: BCOP172
order: 172
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-12T22:13:26Z'
created-by: agent
priority: P1
work: S
---

# PVPS06: validate normalized full-vocab rows before near-lossless scoring

## Discovery / change

PVPS06 was implemented 2026-09-25 (ca39868feb8b) but its plan remained stale. In full_vocab.py, dense_logprobs validates complete finite token IDs without checking normalized probability mass; step_metrics clamps negative KL to zero. A host exact-formula test accepted a +0.01 common subject-logprob shift even though total probability mass became 1.01005. Four host assertions passed. This is a conditional malformed-input false accept, not a reproduced server/GPU fault or measured performance gain.

## Owner / recent independent work

PVPS06 owns the existing comparator; the 1207 producer already uses NEAR_LOSSLESS and 1215 uses BIT_IDENTICAL. The latest PVPS06 implementation commit is from September 25; no PVPS06 work or queued experiment was found within the preceding 12 hours on inspected branches. Protected RAD01 patch-registry migration and Radiance Flash-Next/MTP/quality/prefill/MoE/collective development were not modified. PEC02/PVPS15 trigger evidence and PVPS14 checkpoint evidence remain separate owners.

## Bounded next action / terminal disposition

In dense_logprobs, add stable max-shifted logsumexp normalization validation for both arms before scoring, with a pinned-server-calibrated versioned tolerance and FullVocabError on invalid rows. Do not silently renormalize, add a second comparator, reinterpret old evidence, or change v1 thresholds. Test both offset signs, valid tail-only drift, large vocab, MTP multirow, bit identity, and error-to-invalid propagation. Accept only with zero false accepts and unchanged valid-row verdicts; block distribution qualification if pinned valid rows cannot be characterized. 1207 remains rejected on token divergence at step 55. No GPU campaign queued.

## Evidence

Host discriminator 4/4 assertions passed; synthetic 151936-token sort median 0.0379 seconds (five host readings). llama.cpp issues #27972/#29456 and vLLM/SGLang normalization source inspected; external results are not BigCherry measurements. No repository pytest, C++/HIP build, server or GPU test ran. Full design and source references are in PVPS06.
