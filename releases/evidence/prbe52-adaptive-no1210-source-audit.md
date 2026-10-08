# PRBE52 adaptive MTP without 1210 — source audit (2026-10-08)

## Hardware input

Owner-provided Brutus Flash-Next IQ4_XS results at 245760 ctx:
- adaptive floor 2: ~-13% at 8K, ~+9% at 98K; output varied between repeats;
- floor 3: near-neutral at 8K/98K and matched incumbent text;
- released build without 1210/1255/1268: 85-86 t/s at 8K, 69-70 t/s at 98K, acceptance 343/503;
- restored adaptive experiment with adaptive off: ~56-57 t/s at 98K, acceptance 320/571.

These measurements motivate the change but are not reproduced by this source-only slice.

## Source-verified

1. 1210 is not a code/anchor dependency of 1268. 1268 edits common speculative/server plumbing only. The old manifest edge was a historical correctness prerequisite.
2. 1210 rewrites decode-scale verify kernel selection:
   - CUDA MMVF dispatch normalizes batches <=8 to the decode-size decision;
   - the separate MMVF fusion-decision call is normalized the same way;
   - llamafile SGEMM is disabled for n<=8;
   - AMD flash-attention WMMA is disabled for n_q<=8 and the decode tile/nwarps/nbatch configuration is forced;
   - RDNA3/RDNA4 MMVQ nwarps use the decode whitelist for batches 1..8.
3. 1210 therefore removes some verify-batch-specific kernel/launch choices. It does not disable batching globally and does not unconditionally disable fusion; it makes the fusion eligibility decision use decode-size semantics.
4. 1295 QSA gather remains active for n_tokens<=8. 1210 does not edit qwen4exp QSA gather code, although gathered attention still reaches flash-attention code whose downstream kernel choice 1210 changes.
5. 1322 look-ahead/promotion is not edited by 1210. Its promotion requires full-front acceptance, so target-logit/acceptance changes can indirectly change promotion rate.
6. 1268's previous BIGCHERRY_MTP_AHEAD rejection hid a bookkeeping bug: an ahead draft calls MTP draft() before target acceptance and would reset last_n_draft. This branch removes that mutable draft-call accounting and derives the actual verified front from verify_h_rows-1.
7. MTP begin() resets controller state per request. 1255 has no clock, address, allocator, or other time/layout input. No source path was found for controller-internal nondeterminism.
8. Server speculative-init exceptions previously logged and fell back to spec=null for ordinary speculation. 1268 now makes that startup failure fatal.
9. 1322 obtains its ahead budget through server_slot::get_n_draft_max(). 1268 now caps that value by the controller's effective per-sequence depth; forced front/tail and fresh draft therefore share one round budget.

## Inference pending hardware

- The ~18% 98K loss is consistent with 1210 forcing decode-style kernel/launch choices on verify batches, where batched/tiled/WMMA paths can be faster. Source proves the path changes, not the magnitude.
- Lower acceptance and changed text are consistent with changed floating-point accumulation/kernel order moving greedy logits near decision boundaries. Source does not imply acceptance must decrease.
- Adaptive repeat divergence can arise as feedback amplification: one backend/path-dependent accept difference changes depth, then verify shape, then subsequent logits. The controller arithmetic itself is deterministic.
- 1210 remaining in the stack does not prove full production-set bit identity: its closure covers selected MMVF/MMVQ/SGEMM/fattn decisions, while other batch-shape-sensitive production paths can still exist.

## New controller hypothesis

1255 now starts at depth 3 (clamped to floor/cap) and evaluates acceptance in deterministic 32-drafted-token windows: <=60% drops one level, >=72% climbs one level, otherwise hold. The intent is to keep short-context work near the observed neutral depth 3 while allowing the ~56% long-context regime to fall to depth 2. This is unverified until the requested same-session 8K/24K/98K A/B.
