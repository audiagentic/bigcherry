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

**Diagnostic run (2026-09-12, real hardware) -- stale recurrent state is
RULED OUT.** Inserted `llama_memory_clear(llama_get_memory(slot.ctx_tgt),
true)` (the `data=true` form, clearing both metadata and backing buffers)
plus `llama_synchronize(slot.ctx_tgt)` directly after the real
`slot.mem.seq_rm(slot.id, p0, -1)` call site
(`tools/server/server-context.cpp`), as a temporary, uncommitted
diagnostic build. Re-ran the exact A/B/C sequence: turn C's output with
the physical clear active is `" So we must be careful. So we must be
careful. So we must be careful..."` (a degenerate repetition loop) --
still does NOT match the cold reference. This is a **third distinct
failure signature** across the three conditions tested (baseline:
fluent-but-wrong; patched, no clear: garbled+immediate-EOS; patched+clear:
degenerate repetition) -- none match cold.

Per GPT's own discriminator criterion, a physical clear that still fails
(just differently) means stale/uncleared recurrent state ALONE is not
sufficient to explain the failure. **This does not prove recurrent-state
handling is uninvolved** (GPT correction, 2026-09-13 review,
`req_0e0c18c7ac2d470c`) -- only that a full physical clear does not, by
itself, make the warm path match cold. The root cause remains open; a
first-token logits comparison across baseline-cold/baseline-warm/
patched-cold/patched-warm (and patched-warm+physical-clear) is the
concrete next step, now in progress (see below). The diagnostic
instrumentation was not committed (deliberately throwaway); this
patch's own `patch.py` carries only the real `n_past` fix.

**First-token logit diagnostic, three rounds (2026-09-13) -- live-hardware
investigation closed, GPT-reviewed and signed off (`req_9fbe906155494eb1`
/ `req_ffd9cf54393e408e`).** Built fresh baseline/patched `llama-server`
binaries at the current pin (`b10901`/`28ff0958291c`).

Round 1 used a freshly reconstructed A/B/C prompt sequence (different
literal text, same shape) -- clean on all 4 conditions (baseline/patched
x cold/warm), no garbling. Round 2 recovered the ORIGINAL exact request
`prompt` fields from the earlier investigation's saved response JSONs
(`turnA_1_fixed.json`/`turnB_fixed.json`/`turnC_cached_fixed.json` on the
Brutus scratch tree each carry a `prompt` field, not just `content`) and
reran those exact prompts -- still clean on all 4 conditions. Round 3 ran
GPT's own designed discriminator (baseline/patched x cold-cacheram0 /
warm-cacheram0 / warm-cacheram512, the exact original prompts): all 6
conditions produced identical, correct output. `--cache-ram 0` (which
structurally cannot engage 1005's checkpoint-selection code path at all)
showed the same ~7x prompt-eval speedup as `--cache-ram 512` warm vs
cold -- confirming GPT's prediction that the timing anomaly noted in the
original investigation (task 84's fast "full re-evaluation") is ordinary
runtime/graph warming, not evidence of stale-state corruption.

**Confirmed separately: the original investigation's on-disk materialized
worktree for this exact patch (`materialized/1005-patched`) was later
found genuinely dirty** -- an independent `PatchSourceIsolationError:
patched source tree was modified after materialization` was hit against
that same path earlier in this session (before any of the above reruns;
the stale worktree was deleted and rebuilt clean). This is consistent
with -- though not proof of the specific mechanism for -- the original
investigation's own admission that it inserted temporary, uncommitted
`llama_memory_clear`/`llama_synchronize` diagnostic instrumentation
directly into that worktree area.

**Conclusion (GPT-reviewed wording): live-hardware investigation closed:
no defect reproduced from clean patch-1005 materializations. The earlier
garbled-output result is not attributable to committed patch code
because its source/build environment was subsequently proven modified
outside the recorded materialization. Controlled baseline/patched,
cold/warm, cache-disabled/cache-enabled reruns all produced identical
correct output.** Do not claim the exact contamination mechanism as
proven -- only that the original failure's attribution is invalidated by
confirmed source-tree contamination risk, and the failure is
non-reproducible from clean materializations.

## Disposition

`state` stays `"untested"` (GPT: promoting further would bypass this
project's formal validation path -- no `validation.toml`/Experiment
Contract exists for this patch yet). The real squash defect found this
session (the deleted `n_past` assignment) is fixed and committed. The
live-hardware garbled-output investigation is closed as PASS: no
reproducible defect found across three independent clean-materialization
reruns, including one structural discriminator (`--cache-ram 0`) that
cannot engage this patch's own code path. Formal lifecycle validation
remains pending creation of a validation.toml/Experiment Contract that
encodes this regression sequence (checkpoint invalidation across a
shorter common-prefix turn) as a repeatable, contract-bound check.
