---
id: QFP31
order: 31
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:23.061778+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# MTP draft prompt window: draft head prefills only the last N prompt tokens

## Description

External report (4-GPU Flash-Next build, 2026-10): prefilling the MTP draft head with only the last 2048 prompt tokens improved its reported configuration materially. Those numbers are from different hardware and are only a hypothesis for BigCherry.

At b11402, Qwen4Exp MTP currently catches the draft context up on **every** target prefill sub-batch. On our topology that catch-up executes on the ROCm3 RX 6900 XT while the target prompt runs on the three-card Meta split. For long prompts this means the draft GPU replays the whole prompt even though speculative generation only needs a useful recent draft context.

The lifecycle matters: server `post_decode()` calls `common_speculative_process()` after every target sub-batch, but calls `common_speculative_begin()` only once `SLOT_STATE_DONE_PROMPT` transitions to generation. Therefore `process()` does not know the final prompt length when early chunks arrive. A correct implementation must not use a guessed threshold such as `pos >= prompt_len-window` inside `process()`.

V1 should qualify a **fresh contiguous prompt only**: collect the last N MTP input rows on the host while target prefill proceeds, skip draft catch-up compute for those eligible fresh-prefill chunks, then rebuild only that bounded tail into the draft KV once in `begin()`. Requests whose observed prompt stream does not start at the sequence start or becomes non-contiguous must stay on the existing full catch-up path. Do not silently window prompt-cache/context-shift cases until an explicit prefill-span contract exists.

## What we already have

### b11402

- `common/speculative.cpp::common_speculative_impl_draft_mtp`
  - owns MTP target-to-draft catch-up and draft generation.
- `common_speculative_impl_draft_mtp::process(const common_batch &)`
  - is called after each target batch;
  - reads target `h_nextn` with `llama_get_embeddings_nextn(ctx_tgt)`;
  - pairs token `x_p` with the previous target hidden row using `pending_h` at a chunk boundary and `h_tgt[k-1]` within the chunk;
  - submits that full batch to `ctx_dft` for every MTP head;
  - updates `verify_h` and `pending_h` from the current target batch.
- `common_speculative_impl_draft_mtp::begin(seq_id, prompt)`
  - is invoked after prompt completion by the server;
  - currently only resets the sampler and verifies draft KV reached `N-1`.
- `common_speculative_impl_draft_mtp::draft()`
  - starts from `pending_h` at the final accepted position, so the prompt-window change must leave that value exact.
- `tools/server/server-context.cpp::post_decode`
  - calls `common_speculative_process(spec.get(), batch.view)` before converting `SLOT_STATE_DONE_PROMPT` to generation;
  - then calls `common_speculative_begin(..., slot.prompt.tokens.get_text_tokens())` after the final prompt batch.
- `src/models/qwen4exp.cpp::llama_model_qwen4exp::graph_mtp::graph_mtp`
  - consumes token embeddings plus target hidden input and writes the MTP KV at the supplied absolute positions.
- `llama_memory_seq_rm()`
  - is already used by the MTP driver for chained-head rollback and is the appropriate existing primitive to clear a sequence before rebuilding a bounded tail.

### BigCherry overlap

- `patches/1317_spec_round_timing` (`BIGCHERRY_SPEC_TIMING`) measures serial speculative round phases.
- `patches/1318_mtp_draft_timing` measures MTP draft submit/sync/rest time.
- `patches/1308_qwen4exp_rollback_copy_no_cont` reduces a separate Qwen4Exp rollback/copy cost; it does not limit prompt catch-up.
- `patches/1255_nro06_adaptive_mtp_depth` + `1268_prbe52_adaptive_mtp_wiring` control draft depth, not prompt KV span.

The idea is **not covered**. Existing timing patches can prove whether the long-prompt catch-up is material, but no patch bounds the initial MTP prompt replay.

## Steps

1. Gate 0: use 1317/1318 plus GPU timing to separate target prefill from MTP `process()` catch-up at 8K/24K/~98K. If MTP catch-up is immaterial on ROCm3, close the item.
2. Add a fresh-contiguous-prefill tail collector behind a default-off flag.
3. During eligible prompt processing, retain only the last N tuples required to reconstruct MTP input; continue updating target verification state exactly as today.
4. In `begin()`, clear the draft sequence and replay only the retained tail at original absolute positions, bounded by the draft context's normal batch/ubatch limits.
5. Keep all non-eligible streams on the current full catch-up path.
6. Validate mechanics, activation, acceptance, target identity and fully separated ABBA.

## Detailed Solution & Technical Design

### Qualification flag and eligibility

New mechanism: `BIGCHERRY_MTP_PROMPT_WINDOW=<tokens>`, default `0` (disabled). First qualification value: 2048 because it is the external hypothesis, not because it is assumed optimal.

Apply only to `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` when:

- `!is_mem_shared`; shared-memory MTP already has no catch-up decode to remove;
- the observed sequence starts at the expected fresh prompt start;
- positions remain contiguous for that sequence through prompt processing;
- input is token-only, matching the current Qwen4Exp MTP restriction;
- window > 0 and the prompt exceeds the window.

Fail closed to the current path if these conditions are not established. Do not window a target prompt-cache hit, context shift, multimodal embedding stream or sparse/non-contiguous sequence in v1.

### Tail data layout

For each sequence keep a bounded ring/deque of at most N MTP input tuples:

```text
token id
absolute llama_pos
h_prev[n_embd]   // exact hidden row that current process() would pair with this token
```

The key is storing `h_prev`, not merely the current target hidden row. Existing `process()` defines the pairing:

- first token in a target sub-batch uses `pending_h[seq]`;
- token k>first uses target `h_nextn[k-1]`.

Build the tuple before updating `pending_h`, then evict from the front when the ring exceeds N. Thus the first retained token still has the correct hidden input from position `tail_start-1` even though that older token is not retained in draft KV.

Continue populating `verify_h` and final `pending_h` exactly as the current path. The target verifier and future `draft()` therefore see unchanged host state.

Memory cost is bounded by `N * n_embd * sizeof(float)` per active sequence plus ids/positions. Record the actual MiB for Qwen4Exp before promotion; do not trade an unbounded CPU allocation for GPU time.

### Deferred draft catch-up

For an eligible fresh prompt, `process()` does **not** call `llama_process(ctx_dft, ...)` for prompt catch-up. It only collects the bounded tail and performs the existing target-hidden/verification bookkeeping.

At `begin(seq_id, prompt)`, after the final prompt batch has been processed:

1. require a valid contiguous collector ending at the prompt's final absolute position;
2. clear that sequence's draft KV with the existing memory API;
3. replay retained tuples into `ctx_dft` with their original absolute positions;
4. split replay into legal draft batch/ubatch sizes rather than assuming N fits one submission;
5. for chained MTP heads, rebuild the same retained window for each head using the current head-selection/sequence-clear rules;
6. restore head 0 and discard the collector.

The draft attention sees only the recent N KV cells, but positions/RoPE remain absolute. This is intentionally a semantic change to the **proposal model context**, not to target verification. It may alter draft tokens/acceptance while target output remains exact.

### Short prompts

If prompt length <= N, preserve current semantics rather than clear/rebuild needlessly: either use normal catch-up throughout or, if qualification implementation defers from the start, replay the complete collected prompt. Benchmarking must show no short-prompt regression.

### Meta split, architectures and fusion

The target three-card Meta split is untouched: no split-state, attention-share, expert-share or collective change. The optimization operates in the draft context and must work regardless of which device(s) back that context.

Host control flow is architecture-neutral. Qualification must include gfx1030 (our RX 6900 XT drafter); if the built-in MTP model places draft work on gfx1100/gfx1201, the same path applies. No architecture-specific dispatch is required.

No fusion contract changes. 1311/1313/1308 and target QSA/MoE fusion continue to see the same graph shapes for each submitted draft batch.

## Code Samples & Guidance

Primary anchors in `common/speculative.cpp`:

- fields around `pending_h`, `i_batch_beg`, `i_batch_end`;
- `common_speculative_impl_draft_mtp::process()` around the comment `pair each token with the tgt embedding shifted right by one position`;
- the per-head `llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get())`;
- `common_speculative_impl_draft_mtp::begin()` around the current `pos_max < N - 1` check.

Do not move generic server lifecycle calls in v1.

Suggested activation evidence:

`BIGCHERRY_PATCH_HIT 1344 mtp_prompt_window seq=<id> prompt=<P> replay=<R> skipped=<P-R>`

Emit once per qualified prompt, never per chunk.

Flag: `BIGCHERRY_MTP_PROMPT_WINDOW`, integer token count, default 0. Invalid/negative values disable the experiment; no compatibility alias.

## Files

New mechanism, so if Gate 0 is positive create:

- `patches/1344_mtp_prompt_window/patch.py`;
- its `README.md` and `SUMMARY.md`;
- `tools/tests/patch/test_1344_mtp_prompt_window.py`;
- production file patched: `common/speculative.cpp`.

Use existing 1317/1318 for timing; do not modify them merely to own this mechanism. Use `tools/lab/flash-next/queue-env-ab.sh` for hardware ABBA.

If Gate 0 is negative, create no package and close QFP31 with the evidence.

## Validation

### Offline mechanics

- Patch-lint plus exact/idempotent patch mechanics.
- Flag 0 leaves MTP processing source-equivalent.
- Ring never exceeds N rows.
- Synthetic chunking test proves the retained tuples are identical for the same token stream split at different ubatch boundaries.
- First retained tuple carries the hidden row from the preceding target position.
- Replay preserves absolute positions and legal batch splitting.
- Non-contiguous/non-fresh/multimodal/shared-memory cases do not enter window mode.

### Activation/equivalence

- Marker reports `replay=min(prompt,N)` for a fresh long prompt.
- Draft prompt-decode token count/census falls from P to N.
- Target logits/output are unchanged. Draft proposals are allowed to differ because the draft KV context is intentionally shorter.
- Record acceptance rate/depth; a speed win that collapses acceptance is not promotable.

### Hardware ABBA

Use full process separation through `tools/lab/flash-next/queue-env-ab.sh`.

A: `BIGCHERRY_MTP_PROMPT_WINDOW=0`
B: `BIGCHERRY_MTP_PROMPT_WINDOW=2048`

Run fresh 8K, 24K and ~98K prompts. Record target prefill t/s, MTP catch-up wall/GPU time, total time to first generated token, decode t/s and acceptance. Require greedy target identity.

Also run the 27B built-in-MTP model if its MTP driver is the same non-shared path; otherwise record not applicable rather than inventing a second implementation.

No multi-session contract campaign.

## Effort & Risk

Effort: M.

Risk: medium/high relative to QFP32 because it intentionally truncates draft-model context and changes proposal quality. The main correctness risks are an off-by-one hidden pairing at the window boundary, stale draft KV, treating a cached/non-contiguous prompt as fresh, and chained-head KV inconsistency.

Expected gain on our topology: medium-to-high for 24K/~98K **if** ROCm3 prompt catch-up is on the critical path. At 8K the gain may be small. The 6900 XT is slower than the target cards, so Gate 0 is important.

## Standards

- Target model semantics and target KV remain unchanged.
- Proposal differences are allowed; target greedy output must remain identical.
- No fixed topology, card count or attention split.
- No q4 KV.
- No multimodal/window fallback hidden behind compatibility behavior.
- Default off for qualification.
- Absolute positions are preserved; no position renumbering.
- Prompt-cache/context-shift cases fail closed until explicitly designed.
- External 2048/window performance is a hypothesis only.

## Acceptance Criteria

- Gate 0 attributes material long-prompt time to MTP prompt catch-up, or the item is closed.
- Eligible fresh prompts replay at most N draft tokens with correct previous-hidden pairing and absolute positions.
- Non-eligible streams remain on current semantics.
- Draft KV is cleared/rebuilt without stale prefix cells.
- Target greedy output is identical.
- Acceptance does not materially regress enough to erase end-to-end gain.
- Fully separated 8K/24K/~98K ABBA shows a repeatable time-to-first-token or total-throughput win with activation evidence.
- No target Meta/fusion/collective behavior changes.

## Notes

Ordering: second, after QFP32. It has higher potential gain than most kernel items on very long prompts but greater semantic risk, so first establish whether draft catch-up is actually critical on the RX 6900 XT.

## Change Log

- 2026-10-07T00:39:23.061778+00:00 (created-by): Created by agent
- 2026-10-07: grounded at b11402 and BigCherry head after QFP32; added server-lifecycle constraint, bounded hidden-input tail design, fresh-prefill eligibility, default-off qualification and lightweight ABBA.
