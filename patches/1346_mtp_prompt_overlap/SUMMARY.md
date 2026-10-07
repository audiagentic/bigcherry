# 1346_mtp_prompt_overlap

**Status:** untested
**Plan item:** QFP31

## What it does

Chunk 1 adds a default-off `BIGCHERRY_MTP_PROMPT_TIMING=1` diagnostic and an explicit prompt-start lifecycle hook. For a single-sequence prompt it accumulates MTP `process()` wall time, target NextN fetch/copy time, and draft-context catch-up `llama_process()` wall time, then emits one line at prompt end.

## Why

Gate 0 measured a 17-20% long-prompt prefill penalty from enabling MTP, while the sidecar GPU was kernel-busy for only a small fraction of the fill. This split determines whether QFP31 should first overlap the unchanged draft catch-up with the next target chunk or instead reduce target-side NextN work.

## Scope

Diagnostic only. The flag defaults off. Chunk 1 does not create a worker, skip prompt tokens, change draft KV, change proposals, or alter target model execution. Mixed-sequence batches are deliberately not attributed by the timing diagnostic.
