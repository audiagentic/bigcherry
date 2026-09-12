# 1005 prompt-cache checkpoint selection

This package owns the implementation, metadata, and validation fixture for
the hybrid/recurrent checkpoint-selection backport. The C++ fixture and its
recorded unit-test result are under `validation/fixtures/` (6/6 pass --
validates the reusable-token arithmetic only, not live server behavior).

## Live multi-turn server validation (2026-09-12, Brutus, gfx1100)

Ran the live test this README previously said was still needed: real
`llama-server`, `lfm2.5-8B` (a real recurrent-state architecture per
`llm_arch_supports_rs_rollback` in `llama-arch.cpp`), `--cache-ram 512
--parallel 1`, temp=0/seed=42. Sequence: turn A primes a checkpoint at a
full prompt position; turn C sends a request sharing only a *shorter*
common prefix with that checkpoint (the exact scenario hybrid/recurrent
memory's exact-position-only validity requires this patch to handle
correctly). Compared turn C's cache-assisted completion against a cold
(fresh-process) reference of the identical prompt -- for temp=0
determinism these must match if cache reuse is correct.

**Real bug found and fixed**: the `checkpoint-restore-reset-log-level`
edit's anchor spanned both the `n_past` reassignment line and the
following `SLT_TRC` log line, but its replace-text only supplied the log
line -- `mode="replace"` silently deleted the `n_past` reassignment from
the generated source. Root-caused by `dev-gpt-agent` (session
`ses_3adef4de3bc249b2`, `req_b97d0da804a34f6f`) from source inspection
alone, confirmed directly against the real vendor source, fixed, and
confirmed the fix compiles and lands correctly in the built binary. See
`patch.py`'s inline comment on that edit for the full detail.

**The fix did not resolve the live-test failure -- a separate, deeper,
unresolved issue remains.** Real verbose server trace for the exact same
turn C request, post-fix:

```
erased invalidated context checkpoint (pos_min = 33, pos_max = 33, ..., pos_next = 0, ...)
created context checkpoint 1 of 32 (pos_min = 32, pos_max = 32, ...)
prompt eval time = 22.34 ms / 37 tokens (...)
eval time = 27.71 ms / 8 tokens (...)
```

The checkpoint is correctly invalidated and erased (not incorrectly
restored -- the n_past-deletion bug is NOT the active path in this
reproducer), `pos_next=0` and all 37 prompt tokens are freshly
re-evaluated -- structurally identical to what a cold request does. Yet
generation still produces garbled output and stops after only 8 tokens
(vs. the cold reference's fluent, correct continuation), instead of
matching the cold reference exactly as a correct full reprocess should.

**Do not treat this as a confirmed "recurrent state is not reset on
checkpoint invalidation" finding** -- per GPT's second-round review
(same session, `req_59b2012bc3934ae6`): "the code says it should be
reset; the physical-clear A/B is the shortest test that converts that
hypothesis into a real result." b10705's hybrid memory routes `seq_rm()`
through both recurrent and attention memory, and `build_rs` explicitly
zeroes a freshly-allocated recurrent-state row before use -- there is no
intended code path where full sequence removal clears KV but deliberately
preserves recurrent state. If this is a real bug, it is a defect in that
reset/`rs_zero`/graph-reuse mechanism itself (`llama-memory-recurrent.cpp`
/ graph state setup), not a gap in patch 1005's own server-side
checkpoint/cache-selection scope.

**Concrete next diagnostic step** (not yet done): in the exact
do-reset/no-valid-checkpoint path, temporarily call
`llama_memory_clear(llama_get_memory(ctx_tgt), true)` (the `data=true`
form, which explicitly clears both metadata and backing buffers) plus
`llama_synchronize(ctx_tgt)` before re-evaluating, then re-run the A/B/C
sequence:
- If C now matches the cold reference: confirms a real defect in logical
  `seq_rm()` -> fresh recurrent-state reconstruction. Open a dedicated
  upstream-level investigation there (out of this patch's scope).
- If C still garbles/hits early EOS: stale recurrent state is effectively
  ruled out; the next target is warm-vs-cold graph/sampler/runtime state,
  starting with first-token logits.

## Disposition

`state` stays `"untested"`. Live validation failed even after correcting
the real squash defect found this session (the deleted `n_past`
assignment). Checkpoint restore is confirmed NOT involved in this specific
reproducer -- root cause of the remaining garbled/early-EOS behavior is
genuinely unresolved and requires the physical-clear A/B diagnostic above
before any further conclusion, positive or negative, is drawn.
