---
id: QFP31
order: 31
plan: patching-qwen-flash-next
state: completed
created-at: '2026-10-07T00:39:23.061778+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# MTP draft prompt window: draft head prefills only the last N prompt tokens

## Description

External report (4-GPU Flash-Next build, 2026-10): prefilling the MTP draft head with only the last 2048 prompt tokens improved its reported configuration materially. Those numbers are from different hardware and are only a hypothesis for BigCherry.

At b11402, Qwen4Exp MTP currently catches the draft context up on **every** target prefill sub-batch. On our topology that catch-up executes on the ROCm3 RX 6900 XT while the target prompt runs on the three-card Meta split. For long prompts this means the draft GPU replays the whole prompt even though speculative generation only needs a useful recent draft context.

The lifecycle matters: server `post_decode()` calls `common_speculative_process()` after every target sub-batch, but calls `common_speculative_begin()` only once `SLOT_STATE_DONE_PROMPT` transitions to generation. Therefore `process()` does not know the final prompt length when early chunks arrive. A correct implementation must not use a guessed threshold such as `pos >= prompt_len-window` inside `process()`.

V1 should qualify a **fresh contiguous prompt only**: collect the last N MTP input rows on the host while target prefill proceeds, skip draft catch-up compute for those eligible fresh-prefill chunks, then rebuild only that bounded tail into the draft KV once in `begin()`. Requests whose observed prompt stream does not start at the sequence start or becomes non-contiguous must stay on the existing full catch-up path. Do not silently window prompt-cache/context-shift cases until an explicit prefill-span contract exists.

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

2026-10-07 Gate 0 POSITIVE and much larger than estimated. ABBA qfp31-gate0 on build b-metamem-mw3 (production with 1343/1344/1345), ctx 245760, A = production with the MTP sidecar drafter, B = NO_MTP=1 (no draft context at all): prefill 24K 1123.2, 1114.1 vs 1341.7, 1340.7 t/s (+20%); 98K 1016.0, 1031.0 vs 1276.7, 1277.9 t/s (+25%). So everything MTP adds during the prompt - the target's NextN outputs for every prompt token plus the draft context's catch-up on the 6900 XT - costs 17-20% of prefill wall time. The same run shows why the draft is kept for decode: 75 vs 40 t/s at 24K, 60 vs 36 at 98K. The prefill kernel profile (gate0-d24576) has the 6900 XT busy only 2.9 s in a 99 s fill, so most of the cost is not GPU kernel time on the drafter: it is serialised host/transfer work per chunk and/or the target-side NextN extraction. Not yet split between target side and draft side. This is the largest prefill lever found so far. Two ways to recover it: (a) the window of this item (draft and NextN work only for the last N prompt tokens), (b) keep the full catch-up but overlap it with the target's next chunk on a worker thread (threads are allowed now; the draft runs on its own GPU) - (b) leaves the draft's context and acceptance unchanged.

2026-10-07 chunk 1 (patches/1346_mtp_prompt_overlap, BIGCHERRY_MTP_PROMPT_TIMING=1, build b-metamem-mp24, runs metamem-mp24 / metamem-mp98). 38,673-token prompt at 1169.5 t/s (33.07 s, 77 chunks): target_nextn_ms=25134.9 draft_process_ms=27328.9 draft_decode_ms=2193.2. 123,782-token prompt at 1071.1 t/s (115.56 s, 243 chunks): target_nextn_ms=83321.1 draft_process_ms=94757.4 draft_decode_ms=11433.8. Reading: target_nextn is 72-76% of the whole prompt wall, i.e. it is where the host first blocks on the target GPUs (llama_process returns with the compute still running; the first NextN getter synchronises) - mostly the target's own compute, not a cost. The draft context's catch-up decode is the directly attributable draft cost: 6.6% of the wall at 38K, 9.9% at 124K (28 -> 47 ms per chunk). The rest of the 17-20% MTP prompt cost (about 10 points) is in none of the counters; hypothesis: the forced synchronise per chunk removes the overlap of host preparation for chunk k+1 with GPU compute of chunk k, plus the NextN extraction. So the overlap worker as first designed recovers at most 6.6-9.9%; the window (no getter, no catch-up, no per-chunk synchronise outside the last N tokens) is the mechanism that can recover the rest. Sent to GPT as chunk 2a (split sync from fetch, host gap counter) and 2b (window first, then overlap inside the window).

2026-10-08 ROOT CAUSE measured (experiment prefill-diag, BIGCHERRY_SUBMIT_TIMING=1, 38,673-token prompt, 76 chunks, runs metamem-pd-mtp / metamem-pd-plain, table by tools/lab/flash-next/submit-timing-table.py). Without a draft context: 1444.6 t/s, 354 ms per chunk; target host time per chunk graph build 16.8 ms, set_inputs 4.6 ms, CPU split 4.5 ms, Meta split input stage 285.5 ms (the wait for the previous chunk's device work) and submit 31.2 ms (subgraph rebuild 9.2, launch 20.2 for 96 subgraphs, AllReduce enqueue 1.8): the host prepares chunk k+1 while the GPUs compute chunk k. With the MTP draft: 1170.8 t/s, 437 ms per chunk (+83 ms); the wait moves into the NextN getter (target_sync 328 ms per chunk), and after it the target GPUs are idle while the host runs in series the fetch (1.8 ms), the draft catch-up (28.9 ms) and the next chunk's graph build (17.4), set_inputs (4.4), CPU split (4.4), subgraph rebuild (9.8) and launches. So the whole MTP prompt cost is target-GPU idle time caused by common_speculative_process() synchronising the target right after every chunk. The window as pushed (1346 chunks 2b-4) never armed in production (needs an empty prompt cache and disabled checkpoints): inert, not measured. New mechanism handed to GPT: deferred catch-up - process(tgt,k+1) first, then the catch-up of chunk k from a NextN buffer that survives the next submit, while the target computes k+1; same data and order for the draft, no thread, no window; expected up to +19% prefill with identical text and acceptance.

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

## Change Log

- 2026-10-07T00:39:23.061778+00:00 (created-by): Created by agent
- 2026-10-07: grounded at b11402 and BigCherry head after QFP32; added server-lifecycle constraint, bounded hidden-input tail design, fresh-prefill eligibility, default-off qualification and lightweight ABBA.

## Code-level review (2026-10-07)

### 1. Verified facts and corrections

Checked against llama.cpp \`d89651a7b205\` and the current validated-enhancements composition.

- The production driver is \`common/speculative.cpp::common_speculative_impl_draft_mtp\`. Its state includes:
  \`std::vector<std::vector<float>> pending_h;\`,
  \`std::vector<int32_t> i_batch_beg/i_batch_end;\`,
  and \`verify_h/verify_h_rows\`.
- The constructor calls
  \`llama_set_embeddings_nextn(ctx_tgt, true, false)\` and
  \`llama_set_embeddings_nextn(ctx_dft, true, true)\`, then sets
  \`is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt\` and
  \`chain_heads = n_mtp_layers > 1 && !is_mem_shared\`.
- The exact prompt catch-up is \`bool process(const common_batch & batch_in) override\`. For non-shared MTP it creates each draft row with
  \`batch.add(batch_in.tokens[k].id, batch_in.tokens[k].pos[0], seq_id, false)\`
  and pairs the hidden input as
  \`k == i_batch_beg[seq_id] ? pending_h[seq_id].data() : h_tgt + (k - 1)*n_embd\`.
  It then calls
  \`llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get())\`
  for each MTP head.
- After catch-up, \`process()\` copies the target \`h_nextn\` rows into \`verify_h\` and copies the final row into \`pending_h\`. A window implementation must perform this bookkeeping unchanged even when it suppresses draft decode.
- \`void begin(llama_seq_id seq_id, const llama_tokens & prompt) override\` is currently only sampler reset plus the \`ctx_dft pos_max < N-1\` diagnostic. It runs after prompt completion.
- Server lifecycle is as described in the old plan: \`tools/server/server-context.cpp\` calls
  \`common_speculative_process(spec.get(), batch.view)\` immediately after every successful target sub-batch, before \`post_decode()\`; only when a slot is \`SLOT_STATE_DONE_PROMPT\` does \`post_decode()\` change it to generating and call
  \`common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens())\`.
- The server already knows the missing qualification information **before the first prompt sub-batch**. In the \`SLOT_STATE_STARTED\` path it computes final \`n_past\` after prompt-cache/chunk-reuse/checkpoint decisions, sets
  \`slot.stats.n_prompt_cached = n_past;\`
  and then
  \`slot.prompt.tokens.keep_first(n_past);\`.
  It also knows total \`slot.task->n_tokens()\`, whether the server has \`mctx\`, and whether \`slot.task->n_tokens_shared > 0\`.
- **Major correction/blocker in the old design:** a bounded ring cannot promise “late failure falls back to full catch-up.” Once more than N skipped rows have been evicted, a later multimodal/non-contiguous/cache invalidation cannot reconstruct those skipped draft rows. The collector-only design is therefore unsafe without an explicit prompt-start contract.
- That contract can be added cleanly at the server point above: only arm windowing when the server has already resolved \`n_past == 0\`, text-only/no-\`mctx\`, no shared-prefix child flow, and total prompt length > window. Then \`process()\` validates the promised contiguous positions as an invariant rather than trying to infer future prompt shape.
- Qwen4Exp's MTP graph is single-head: \`src/models/qwen4exp.cpp::graph_mtp::graph_mtp\` asserts \`hparams.n_layer_nextn == 1\`. V1 should therefore require \`n_mtp_layers == 1 && !chain_heads && !is_mem_shared\`; generic chained-head support is unnecessary for these two Qwen3.8 targets and materially complicates restore semantics.
- \`graph_mtp\` creates \`build_inp_pos()\`, consumes target-hidden input, runs its attention/FFN block, and writes \`h_nextn\`. The source comment at its memory setup says “the draft memory has no recurrent layer, but its input still has to be allocated.” This removes the main recurrent-state objection for production Qwen4Exp.
- \`llama_memory_seq_rm(mem, seq_id, -1, -1)\` is the correct whole-sequence clear: the public API explicitly states that removing a whole sequence never fails. For a window replay, use this full clear; partial removal is unnecessary.
- Absolute positions must remain the original \`batch_in.tokens[k].pos[0]\`; do not renumber the retained tail to zero.
- The old package number is out of date: \`1344\` is now \`1344_dsv4_hc_grid_index\`. The next package for this item is \`1345_mtp_prompt_window\`.

### 2. Composition and post-patch anchors

In \`[patch-set.validated-enhancements]\`, no current production patch edits \`common/speculative.cpp::common_speculative_impl_draft_mtp::{process,begin}\` or the server's \`SLOT_STATE_STARTED\` prompt-cache block. The primary anchors are therefore still upstream text at this composition.

Production interactions:

- \`1343_mtp_nextn_rereserve\`
  - \`mtp-rereserve-include\`
  - \`mtp-rereserve-setter\`
  edits \`src/llama-context.cpp::llama_context::set_embeddings_nextn\`, not the MTP driver. It is nevertheless essential composition: Gate 0 must be measured **with 1343 enabled**, because it already removed per-prefill-chunk scheduler replans and reduced the opportunity that QFP31 can claim.
- \`1297_draft_vocab_trim::mtp-trimmed-head\` edits \`src/models/qwen4exp.cpp::graph_mtp\`. Its own comment distinguishes “batches with no output rows - MTP prompt replay”; QFP31 replay rows must keep \`output=false\` exactly as current \`process()\` does so 1297's prompt path is unchanged.
- \`1308_qwen4exp_rollback_copy_no_cont::rollback-no-cont\`, \`1311_hc_pre_q81\`, \`1313_scale_act_fuse\`, and \`1344_dsv4_hc_grid_index\` affect kernels/graph execution inside Qwen4Exp but not MTP lifecycle anchors.
- \`1281_moe_mul_mat_id_range\` / \`1283_qwen4exp_expert_parallel\` do not edit the driver. They matter only if a draft context itself uses the corresponding Meta/EP composition.

Non-production diagnostics/optional patches that do touch these areas:
- \`1255_nro06_adaptive_mtp_depth::adaptive-controller\` inserts immediately before the MTP struct.
- \`1268_prbe52_adaptive_mtp_wiring\` edits MTP members/constructor/begin/draft/accept, including \`prbe52-state\`, \`prbe52-ctor\`, and \`prbe52-begin-reset\`.
- \`1315_mtp_draft_trace\` edits draft/accept only.
- \`1318_mtp_draft_timing\` edits the draft loop only.
- \`1317_spec_round_timing\` edits the server's \`common_speculative_process\` call site as \`spec-timing-process\`.
These are not in the requested production validated set, but Gate-0 diagnostic builds may include them. QFP31 anchors must therefore avoid their modified ranges where possible.

Post-composition anchors for 1345:
- public API declaration: \`common/speculative.h\`, immediately after exact
  \`void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);\`.
- generic virtual: \`common/speculative.cpp::common_speculative_impl\`, after
  \`virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;\`.
- MTP state: anchor after the composed \`pending_h\` declaration, not on the optional 1268-expanded block.
- prompt-start server contract: anchor after exact
  \`slot.prompt.tokens.keep_first(n_past);\`
  in the \`SLOT_STATE_STARTED\` block; 1317 does not touch this region.
- process collector: anchor on the exact comment
  \`// pair each token with the tgt embedding shifted right by one position\`
  and the current \`batch.add(..., false)\` / h-row pair, not the outer \`if (!is_mem_shared)\`.
- replay: anchor in the MTP-specific \`begin()\` after \`common_sampler_reset(smpls[seq_id].get());\`; if 1268 is intentionally composed in another recipe, anchor **after** its \`prbe52-begin-reset\` insertion.

### 3. Gaps and risks

- **Prompt-start contract is mandatory.** Do not code the old “infer freshness in process() forever” design. The server has final \`n_past\` and total prompt length before processing; use it. Without this, late invalidation after ring eviction is unrecoverable.
- V1 eligibility should be exactly: window > 0, non-shared single-head MTP, server contract says \`n_past == 0\`, no MTMD server context, no shared-prefix child prompt, total prompt length > window. Everything else stays on native catch-up from the first row.
- After arming, every observed row for that sequence must have the expected next absolute position and token input. A mismatch is a violated server/driver invariant: disable/error before using a partially reconstructed draft context; never silently continue with missing rows.
- Multi-slot continuous batching is per-sequence. One qualified sequence may be deferred while another sequence in the same \`common_batch\` still needs native catch-up. Build the immediate draft batch only from non-window rows; do not skip the whole batch globally.
- Store **the hidden input actually used by current process()**, not current target hidden output: first row of each source batch uses pre-update \`pending_h\`; later rows use \`h_tgt[k-1]\`.
- Use a contiguous bounded ring per sequence (\`token[N]\`, \`pos[N]\`, \`h_prev[N*n_embd]\`) rather than N heap-allocated row vectors. Check \`N*n_embd*sizeof(float)\` for overflow and record actual MiB at activation.
- Short prompts must never arm because the server already knows final prompt length. This is better than the old defer-then-replay-short design and keeps short-prompt semantics/source schedule unchanged.
- Qwen4Exp production is single-head; fail closed on \`chain_heads\` rather than designing a generic multi-head rebuild now.
- Meta: the target three-card Meta graph is untouched. The sidecar draft context is on gfx1030; the 27B built-in-MTP lane may place its draft context differently. Use context/device discovery only, never ordinals in code.
- 1281/1283: no op-param or range semantics change. If a future draft context uses EP, the replay's changed token dimension will reach range MMQ normally; preserve exact ids and zero semantics.
- Q8_1 cache (1235/1307/1309-1312): replay changes draft graph batch timing/shape and therefore cache generation/capture behavior, but it must not add a separate Q8_1 path. Record cache activation/hit evidence in fusion-on probes.
- CUDA graphs: replay at \`begin()\` should use normal \`llama_process\` and chunk by \`llama_n_batch(ctx_dft)\`; the context handles ubatching. Do not assume production 512 in code. Reusing normal batch sizes gives existing captured graph shapes the best chance to hit.
- 1343 should prevent scheduler re-reservation churn from NextN mode, but QFP31 can still alter draft graph-capture warmup/order. Count captures/recaptures in ABBA.
- FKE01 applies despite target graph source being unchanged: changing when/how the draft context allocates and changing accepted proposals can alter subsequent generated graph allocation/fusion admission. Target identity must therefore be established with \`GGML_CUDA_DISABLE_FUSION=1\` on **both** A and B. Fusion-on runs are performance/behavior probes.
- Draft proposals are intentionally allowed to differ; 1315 already documents baseline run-to-run draft near-tie nondeterminism. Do not require bit-identical proposal traces as an acceptance criterion.
- No host worker thread is useful. The host ring copies are required state capture and should remain synchronous with \`process()\`.

### 4. Concrete implementation outline

Create \`patches/1345_mtp_prompt_window\`.

Flag:
\`BIGCHERRY_MTP_PROMPT_WINDOW=<tokens>\`; default \`0\` (off). First experimental value 2048. Reject/disable values <=0 or larger than the draft context; no alias/shim.

Add a minimal lifecycle contract, not a server behavior rewrite.

In \`common/speculative.h\`:

\`void common_speculative_prefill_begin(common_speculative * spec, llama_seq_id seq_id, int32_t n_prompt, int32_t n_cached, bool fresh_text);\`

In \`common_speculative_impl\` add optional virtual:

\`virtual void prefill_begin(llama_seq_id, int32_t, int32_t, bool) {}\`

The public wrapper iterates active implementations, as \`common_speculative_begin/process\` already do.

Server call, once in \`SLOT_STATE_STARTED\` immediately after final \`n_past\` is committed with \`slot.prompt.tokens.keep_first(n_past)\`:

\`common_speculative_prefill_begin(spec.get(), slot.id, slot.task->n_tokens(), n_past, mctx == nullptr && slot.task->n_tokens_shared == 0);\`

The MTP override arms collection only when:
- flag/window positive and \`n_prompt > window\`;
- \`n_cached == 0 && fresh_text\`;
- \`!is_mem_shared && !chain_heads && n_mtp_layers == 1\`.

State per sequence:

\`struct bc_mtp_prompt_tail { bool armed; int32_t expected_prompt; llama_pos expected_pos; size_t head,count; std::vector<llama_token> ids; std::vector<llama_pos> pos; std::vector<float> h_prev; };\`

Allocate lazily on a qualified prompt. Ring capacity is exactly window.

In \`process()\`:
1. build \`i_batch_beg/end\` exactly as native;
2. obtain \`h_tgt\`;
3. for each token:
   - compute the same native \`h_row\` before \`pending_h\` is updated;
   - if that sequence is armed, assert/validate expected absolute position, append \`{id,pos,h_row}\` to the ring and do **not** add it to the draft catch-up batch;
   - otherwise add the row to the ordinary draft batch unchanged;
4. call native draft catch-up only when the immediate batch is non-empty;
5. run native \`verify_h/pending_h\` update unchanged for all sequences.

In MTP \`begin()\`, before the existing pos-max diagnostic:
1. if sequence not armed, execute native behavior unchanged;
2. require collected count == min(prompt_size, window), final expected position == prompt end, and server-advertised prompt size == \`prompt.size()\`;
3. \`GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, -1, -1));\`;
4. replay ring rows in chronological order with **original absolute positions**, \`output=false\`, and stored \`h_prev\`;
5. split submissions at \`llama_n_batch(ctx_dft)\`; let \`llama_process\` do physical ubatching;
6. verify \`llama_memory_seq_pos_max(...) == prompt.size()-1\`, restore no head offset because V1 forbids chain heads;
7. emit marker and clear the collector.

Marker:
\`BIGCHERRY_PATCH_HIT patch=1345_mtp_prompt_window path=prompt_replay seq=<id> prompt=<P> replay=<R> skipped=<P-R> host_mib=<...>\`.

Required patch edits:
- \`mtp-window-api\`: \`common/speculative.h\`, exact \`common_speculative_begin\` declaration; \`insert_after\`.
- \`mtp-window-virtual\`: \`common/speculative.cpp\`, exact base-class \`virtual void begin(...)=0;\`; \`insert_after\`.
- \`mtp-window-wrapper\`: \`common/speculative.cpp\`, insert the public wrapper adjacent to \`common_speculative_begin\`; \`insert_before/after\` exact function signature.
- \`mtp-window-state\`: MTP struct, insert after exact \`pending_h\` declaration; \`insert_after\`.
- \`mtp-window-ctor\`: MTP constructor, insert after exact \`pending_h.assign(...)\`; \`insert_after\`.
- \`mtp-window-prefill-begin\`: insert MTP override immediately before its exact \`void begin(...)\` signature; \`insert_before\`.
- \`mtp-window-process\`: replace only the native row-build + catch-up submission region inside \`if (!is_mem_shared)\`; retain the verify/pending block byte-for-byte after it.
- \`mtp-window-replay\`: extend MTP \`begin()\` immediately after sampler reset (or after composed 1268 reset when that optional recipe is used); \`insert_after\`.
- \`mtp-window-server-contract\`: \`tools/server/server-context.cpp\`, insert after exact \`slot.prompt.tokens.keep_first(n_past);\`; \`insert_after\`.
- Add EnvDoc with default 0.

Offline mechanics must parse/compare the exact composed MTP process/begin text, verify server contract placement after final cache resolution, and exercise ring/chunking logic independently.

### 5. Gate 0 and lightweight validation

Gate 0 must be rerun **after 1343**; earlier prefill numbers that included 246 scheduler replans at 98K are stale.

Use current validated production + diagnostic \`1317_spec_round_timing\` in a single-slot run, with f16 target/draft KV, ub/batch 512, depth 3, normal Meta flags. Run fresh prompts at 2048, 8192, 24576 and ~98304.

For each prompt P:
- record wall time from prompt start to first target sample/TTFT;
- from the first \`BIGCHERRY_SPEC_TIMING\` line, isolate accumulated prompt \`process_us\` by subtracting one steady-state generated-round \`process_us\` (use median rounds 2-6);
- use rocprof Agent_Id to sum gfx1030 kernels during that prompt catch-up interval;
- record target-side wall separately so draft catch-up is not confused with target prefill;
- confirm 1343 reports zero steady prompt replans after its initial reserve.

Define:
\`S(P) = T_mtp_prompt_process(P) - T_mtp_prompt_process(2048)\`.

Gate 0 is positive only if:
- \`S(24576) >= 1.0% * TTFT(24576)\` **or** \`S(98304) >= 1.5% * TTFT(98304)\`; and
- gfx1030 prompt-catch-up GPU busy time explains >=70% of the measured removable \`process_us\` (otherwise investigate host/lifecycle overhead before coding); and
- the baseline MTP path is non-shared, single-head Qwen4Exp as expected.

If the threshold is not met after 1343, close QFP31; do not use the external 2048 result as justification.

Validation after implementation:

- offline exact/idempotent mechanics + patch-lint;
- pure state-machine test: server contract arms only fresh text, n_cached=0, P>N; cached, MTMD, shared-prefix, short, shared-memory, and chained-head cases remain native from row 0;
- chunk-boundary test: identical retained \`{id,pos,h_prev}\` for the same prompt split into 1/7/512/mixed sub-batches;
- marker must report replay=min(P,N), skipped=P-N, correct absolute end position, bounded host MiB;
- draft prompt token census must fall from P to N only on qualified requests;
- fully separated ABBA via \`tools/lab/flash-next/queue-env-ab.sh\`, explicitly \`CTK=f16 CTV=f16 CTKD=f16 CTVD=f16\`, at 8K/24K/~98K:
  A=\`BIGCHERRY_MTP_PROMPT_WINDOW=0\`,
  B=\`BIGCHERRY_MTP_PROMPT_WINDOW=2048\`;
- performance ABBA runs with fusion on; record prompt t/s, TTFT, gfx1030 catch-up time, total draft/accept counts, accepted tokens per round, generation t/s, CUDA graph capture/recapture and Q8_1 cache activation;
- separate identity/control ABBA with \`GGML_CUDA_DISABLE_FUSION=1\` on **both** A and B. Require identical greedy target text/hash; draft proposals may differ by design;
- fusion-on probes required by FKE01; investigate any target-text change rather than attributing it to allowed draft-context truncation;
- require no material net regression in generation: accepted tokens/round should not fall >10% unless total request wall still improves and the owner explicitly re-qualifies the tradeoff;
- run the 27B built-in-MTP lane only if its runtime reports the same non-shared single-head driver; otherwise mark not applicable, not a second implementation;
- never q4 KV.

### 6. Verdict

**GO AFTER GATE 0**, but only with the prompt-start server contract above. The old bounded collector without that contract is not safe to implement.

Expected gain on the production three-card target + gfx1030 sidecar, if Gate 0 passes: **8K +0.0% to +0.4%, 24K +0.4% to +1.3%, ~98K +0.8% to +2.5% prefill/TTFT**. The upper end requires the gfx1030 catch-up to remain materially serialized after 1343.
- 2026-10-07T07:51:46.301374+00:00 (updated-by): Updated: priority='P0', section:notes

## Chunk 1 implementation decision (2026-10-07)

**Decision: OVERLAP first, after the cost split below.** Keep the full prompt catch-up and proposal context, but in chunk 2 move only the draft-context catch-up for prompt chunk `k` onto one persistent worker while the target computes chunk `k+1`. This is lower risk than WINDOW because it preserves every draft KV row, proposal distribution and acceptance opportunity. It can recover only the draft-side/host portion of the measured penalty; target NextN extraction remains serial. Therefore chunk 1 is a stop gate: if `target_nextn_ms` dominates and the overlap-able remainder is small, do not build the worker first; switch QFP31 to WINDOW using the existing prompt-start contract.

Exact b11402 call order to preserve:

1. `tools/server/server-context.cpp`, `SLOT_STATE_STARTED`: prompt-cache/checkpoint decisions finish, final `n_past` is committed, then `slot.prompt.tokens.keep_first(n_past)`. Chunk 1 calls new `common_speculative_prefill_begin(spec, slot.id)` here. No prompt work has started yet.
2. For each rendered target sub-batch, the server calls `llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get())`, synchronizes target output when required, then calls `common_speculative_process(spec.get(), batch.view)`.
3. `common/speculative.cpp::common_speculative_impl_draft_mtp::process()` reads target NextN state, builds the existing shifted-hidden draft catch-up batch, calls `llama_process(ctx_dft, ... DECODE ...)`, then updates `verify_h`, `verify_h_rows` and final `pending_h`.
4. After the final prompt sub-batch, server `post_decode()` transitions the slot to generation and calls `common_speculative_begin(...)`. This is the mandatory future worker flush boundary before generation can read draft state.

Threaded shared-state inventory for chunk 2/3:

- **Main-thread only during prompt:** `ctx_tgt`; target NextN getters/copies; `pending_h`; `verify_h`; `verify_h_rows`; `i_batch_beg/end`. These remain synchronous so target output/verification bookkeeping is unchanged.
- **Worker-only while prompt overlap is armed:** prompt catch-up calls on `ctx_dft`. The worker must own a separate catch-up `common_batch`; it must not concurrently reuse the MTP member `batch`, which `draft()` uses after the prompt.
- **Job payload is immutable/deep-copied:** token ids, absolute positions, sequence ids and the exact hidden-input rows that native `process()` would pass to the draft. Pointers into target NextN buffers may not escape the submitting call because later target processing can reuse those buffers.
- **Generation/sampler state:** `smpls`, `backend_chains`, `i_last`, `chain_h` and draft generation use are main-thread only after a flush. V1 overlap should qualify only non-shared, single-head MTP (`!is_mem_shared && !chain_heads && n_mtp_layers == 1`); this matches production Qwen4Exp and avoids concurrent shared/chained-head KV semantics.
- **Queue bound/order:** FIFO, at most one running plus one queued catch-up job. This preserves prompt order and prevents hidden-row copies growing with prompt length.
- **Required future flushes:** before `begin()`, `draft()`, speculative/draft state save or load, prompt-cache draft save/load, context shift/removal that touches draft memory, cancellation, slot reuse with another prompt, and MTP destruction. Chunk 3 must name and wire every concrete server/driver path.

Server-to-driver contract for OVERLAP is deliberately smaller than WINDOW: one `prefill_begin(seq_id)` after cache/checkpoint resolution and before the first target prompt batch, plus existing `begin()` at prompt end. No prompt length/window or cache count is needed because full catch-up is retained. Prompt-cache/checkpoint restores that happen before `prefill_begin` remain synchronous.

Chunk-1 diagnostic (`patches/1346_mtp_prompt_overlap`, `BIGCHERRY_MTP_PROMPT_TIMING=1`) is behavior-neutral and intended for the single-slot Gate-0 run. It prints once at prompt end:

`BIGCHERRY_MTP_PROMPT_TIMING target_nextn_ms=... draft_process_ms=... draft_decode_ms=... chunks=... tokens=...`

Definitions: `draft_process_ms` is inclusive wall time in MTP `process()`; `draft_decode_ms` is wall time inside its existing prompt catch-up `llama_process(ctx_dft, ...)` calls; `target_nextn_ms` is host-visible wall in target NextN buffer acquisition plus per-row NextN getters/copies, including any synchronization those getters force. The residual `draft_process_ms - target_nextn_ms - draft_decode_ms` is other synchronous MTP host work. This does **not** measure the target graph's internal NextN compute separately; compare with the existing no-MTP Gate 0 for that ceiling. Mixed-sequence batches are intentionally not attributed by this diagnostic.
- 2026-10-07T09:29:23.211096+00:00 (updated-by): Updated: section:notes

## Chunk 2b implementation decision (2026-10-07)

Chunk-1 hardware changes the mechanism order: **WINDOW first; OVERLAP deferred.** The measured
`target_nextn_ms` was 72-76% of whole prompt wall, while draft catch-up decode was 6.6% at
38K and 9.9% at 124K. Code inspection confirms the first target NextN getter is a synchronization
point after asynchronous target `llama_process()`; therefore the original counter mostly observed
the target join, not independent NextN compute. Chunk 2a now reports explicit `target_sync_ms`,
post-sync `target_fetch_ms`, and return-to-next-submit `host_gap_ms`.

`BIGCHERRY_MTP_PROMPT_WINDOW=N` (default 0) qualifies only fresh text prompts with
`n_cached == 0`, `P > N`, non-shared MTP and exactly one trained head. The server passes
`P`, final `n_past`, and the fresh-text eligibility bit at the existing resolved prompt-start
boundary. Qualified early chunks do not call a target NextN getter and do not run draft-context
catch-up. One boundary exception is required for correctness: replay row `p=P-N` consumes the
native input hidden `h[p-1]`, so the chunk containing `P-N-1` is fetched once when `P-N`
starts on a later chunk. It is not draft-decoded.

The collector is simpler than the earlier ring outline because prompt length is now known before
row 0: it preallocates exactly N `{id,pos,h_prev}` rows and fills only absolute positions
`[P-N,P)`. At `common_speculative_begin()`, it clears only that sequence in the draft memory,
replays exactly those N rows at their original absolute positions, synchronizes the draft context
before freeing host input storage, then generation proceeds normally.

At generation start after a qualified replay:
- draft KV contains exactly prompt positions `[P-N,P-1]` for that sequence, at original positions;
- replay row `p` is token `x[p]` paired with the same hidden input native MTP uses,
  `h[p-1]`; the boundary predecessor is captured explicitly;
- `pending_h` is the target hidden row at `P-1`, and `verify_h/verify_h_rows` describe the
  final target verification batch exactly as native `process()` would;
- the bounded host collector has been joined and freed; no pending replay work remains;
- `draft()` seeds `id_last` at `pos0=P` with `pending_h` and attends only to the retained
  absolute-position tail. Target KV, target logits and target sampling are unchanged.

This version intentionally does not add the worker overlap. Re-measure after WINDOW; overlap is
worth adding only if the remaining in-window draft catch-up is still material.

## Chunk 3 lifecycle invariants (2026-10-07)

WINDOW v1 is deliberately conservative outside the fresh-prompt path:

- **Prompt cache reuse:** a completed WINDOW request leaves only a draft KV tail and its
  `pending_h` carry is host-only. If a later request reuses `n_cached > 0`, MTP is suppressed
  for that request rather than inventing the missing boundary carry. A fresh `n_cached == 0`
  request resets row-0 carry and may arm WINDOW again.
- **RAM prompt-cache load:** the cache stores target and draft KV bytes but not MTP
  `pending_h`/WINDOW host state. Successful cache load therefore poisons MTP state; a cached
  request is target-only.
- **Prompt checkpoints:** WINDOW does not arm when `params_base.n_ctx_checkpoints > 0`;
  a checkpoint restore cannot reconstruct the hidden-row collector without serializing it.
- **Short prompts:** `P <= N` never arm and use the native full MTP prompt path.
- **Cancel mid-prompt:** release from STARTED/PROCESSING_PROMPT/DONE_PROMPT poisons and frees the
  incomplete collector before slot reuse.
- **Slot reuse / explicit clear:** `prompt_clear()` clears target and draft memories through
  `common_memory` and resets WINDOW host carry to clean row-0 state.
- **Context shift:** existing `common_memory::seq_rm/seq_add` mutates target and draft memories
  identically; the replayed tail stays position-aligned.
- **Slot state save/load:** slot save/restore does not restore the WINDOW host carry; restore
  poisons MTP state, leaving target state usable while suppressing MTP proposals.
- **Marker:** each successful replay emits one
  `BIGCHERRY_PATCH_HIT patch=1346_mtp_prompt_overlap mechanism=window window=... replay=... skipped=... end_pos=... host_mib=...`
  line after draft replay synchronization and before collector storage is freed.

## Ledger-events

- chg_20261007_133409_the-low-vram-tensor-split-layo_8043
- 2026-10-07T13:34:19.875157+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-07T13:34:30.973714+00:00 (updated-by): Updated: section:notes

## Completion (2026-10-08)

1348_mtp_deferred_catchup qualified and promoted at b11474. Brutus adoption ABBA measured +5.4% / +6.3% / +9.7%
prefill at 8K / 24K / 98K with complete separation, PATCH_HIT activation, unchanged decode/acceptance within noise,
and greedy target identity at all depths. The original 1346 prompt-window mechanism did not qualify and is removed
from that diagnostic package separately.

### Why the gain is ~6-10%, not the ~19% upper bound

The earlier 83 ms / 437 ms estimate was the whole MTP-induced serial gap between target chunk submissions, not 83 ms
of draft compute that this patch could hide. Prior timing split that gap into roughly 29-47 ms/chunk of actual draft
catch-up plus host work needed to prepare and submit the next target chunk.

1348 hides the previous chunk's draft catch-up under the current target chunk, but `process_deferred()` then calls
`llama_get_embeddings_nextn(ctx_tgt)` for the current chunk before returning. That getter synchronizes the current
target, and the host copy of the NextN rows follows it. Therefore the server still cannot build/set inputs/rebuild and
launch the following target chunk while the current target is executing. The remaining serialization is primarily:

- current-chunk target synchronization in the NextN getter;
- the host copy of the current chunk's NextN rows;
- next-chunk graph build / set_inputs / CPU split / Meta subgraph rebuild and launches, which still occur after that
  synchronization instead of overlapping target compute;
- the final pending catch-up flush before speculative begin/sampling (one chunk, proportionally larger at 8K).

The measured +5.4/+6.3/+9.7% therefore matches the previously measured 6.6-9.9% draft-decode share much better than
the 19% whole-gap upper bound.

Follow-up: QFP42 should remove the per-chunk blocking NextN getter from the host critical path. Preferred design is an
MTP-owned asynchronous NextN staging slot: enqueue a device->pinned-host copy after target chunk k on the target
stream(s), return immediately so the host can prepare/submit k+1, then after k+1 is submitted wait only for k's staging
event and run draft catch-up k. This needs explicit output-lifetime/event ownership for Meta/split outputs and must
preserve the same row order/identity. Final flush remains serial. Do not add a worker thread unless the event/staging
path proves insufficient.

