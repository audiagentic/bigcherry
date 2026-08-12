"""Upstream backport: fix hybrid/recurrent memory checkpoint selection and
prompt-cache entry selection.

Cherry-picked and squashed from a 4-commit upstream stack (old-lane patch
pack ``features/prompt-cache-checkpoint-selection/b10227``, re-verified
against this project's current, numerically newer base rather than trusted
blindly -- ``src/llama-memory-hybrid.cpp``'s ``seq_pos_min`` comment was
checked directly and still has the pre-fix wording, proving the fix never
landed on our base despite it being newer than the old lane's b10227 pin):

1. ``server : fix checkpoint handling for hybrid/recurrent models``
   (upstream PR #24055, not #25592 as the old lane's README implied --
   confirmed from the real commit subject line).
2. ``server : drop the unsafe rs rollback fast-path and adopt restored
   checkpoints``.
3. ``server: fix prompt cache entry selection and f_keep filter``.
4. ``server: correctly reuse full prompt without checkpoints`` (amends
   commit 3's own ``server_prompt_cache_n_reusable`` before it shipped).

Squashed into final-state edits against our current tree rather than
replayed commit-by-commit, since commits 2 and 4 each rewrite code
commits 1 and 3 just introduced -- replaying intermediate states that
never exist in our tree would just be extra anchors that can drift.

Root problem: hybrid/recurrent memory (e.g. GatedDeltaNet/Mamba-style
models) is only valid at its *exact* final position -- unlike SWA memory,
which is valid over a range. The pre-fix code treated every checkpoint as
range-valid, so hybrid/recurrent checkpoints could be selected and
restored incorrectly. Fixed by adding ``ctx_tgt_state_exact()`` and gating
checkpoint creation/selection/invalidation on it. Separately, prompt-cache
entry selection (``server_prompt_cache::load``) rejected any cached entry
below 25% keep-ratio even when a hybrid/recurrent checkpoint inside it
could still salvage a large prefix, and required a candidate to beat the
*current slot's* f_keep -- trivially true whenever the slot holds a small
unrelated prompt -- rather than just picking whichever entry saves the
most work.

Not gated to CUDA/HIP at all -- this is server-side C++ with no GPU
backend dependency, applies identically to every build.

The anchors below deliberately use executable tokens only. The patcher blanks
comments and C/C++ string literals before matching, so log messages and prose
must never be part of an anchor.
"""

from bigcherry.patcher import Edit, FilePatch

GROUP = "upstream-fixes"

SERVER_CONTEXT_PATCH = FilePatch(
    path="tools/server/server-context.cpp",
    description="Fix hybrid/recurrent context-checkpoint creation, selection "
                "and invalidation to require exact-position validity instead "
                "of treating every checkpoint as range-valid like SWA "
                "(upstream PR #24055 + follow-up commit)",
    edits=(
        Edit(
            id="ctx-tgt-state-exact-and-checkpoint-adopt",
            anchor=(
                r"    void create_checkpoint\(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max\) \{\n"
                r"        const int id_task = slot\.task->id;\n"
            ),
            rationale="create_checkpoint's opening -- add the exact-state helper and adopt-existing-checkpoint fast path",
            mode="replace",
            text=(
                "    // whether the memory state is valid only at its exact final position (hybrid/recurrent),\n"
                "    // as opposed to a range of positions (SWA)\n"
                "    bool ctx_tgt_state_exact() const {\n"
                "        return ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||\n"
                "               ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS;\n"
                "    }\n"
                "\n"
                "    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {\n"
                "        const int id_task = slot.task->id;\n"
                "\n"
                "        // an equivalent checkpoint already exists (e.g. it was just restored)\n"
                "        if (!slot.prompt.checkpoints.empty() &&\n"
                "                slot.prompt.checkpoints.back().n_tokens == slot.prompt.n_tokens() - n_tokens_cur &&\n"
                "                slot.prompt.checkpoints.back().pos_max == pos_max) {\n"
                "            // adopt the checkpoint so the min-step eviction below does not erase it\n"
                "            slot.prompt.checkpoints.back().id_task = id_task;\n"
                "            return;\n"
                "        }"
            ),
            guard="ctx_tgt_state_exact",
        ),
        Edit(
            id="create-checkpoint-pos-min-exact-clamp",
            anchor=r"        cur\.update_pos\(slot\.prompt\.n_tokens\(\) - n_tokens_cur, pos_min, pos_max\);",
            rationale="clamp pos_min to pos_max for exact-position (hybrid/recurrent) memory before storing the checkpoint",
            mode="insert_before",
            text=(
                "        // the state of hybrid/recurrent memory is valid only at its exact final position\n"
                "        if (ctx_tgt_state_exact()) {\n"
                "            pos_min = pos_max;\n"
                "        }\n"
                "\n"
            ),
            guard=r"if \(ctx_tgt_state_exact\(\)\) \{\n            pos_min = pos_max;",
        ),
        Edit(
            id="create-checkpoint-log-level",
            anchor=(
                r"        common_speculative_get_state\(spec\.get\(\), slot\.id, cur\.data_spec\);\n\n"
                r"        SLT_TRC\(slot,"
            ),
            rationale="promote the checkpoint-creation log from TRC to INF (matches upstream's other checkpoint log promotions in this same fix)",
            mode="replace",
            text=(
                "        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);\n\n"
                "        SLT_INF(slot,"
            ),
            guard=r"SLT_INF\(slot,\n                \"created context checkpoint",
        ),
        Edit(
            id="remember-divergence-pos-next-lcp",
            anchor=r"                            llama_pos pos_next = slot\.prompt\.tokens\.pos_next\(n_past\);",
            rationale="stash the pre-restore divergence position for the checkpoint-invalidation pass below",
            mode="insert_after",
            text=(
                "\n"
                "\n"
                "                            // pos_next can be reduced below by a checkpoint restore - remember the\n"
                "                            // divergence point for the checkpoint invalidation\n"
                "                            const llama_pos pos_next_lcp = pos_next;"
            ),
            guard=r"const llama_pos pos_next_lcp = pos_next;",
        ),
        Edit(
            id="checkpoint-search-exact-vs-range",
            anchor=(
                r"if \(pos_min >= pos_min_thold\) \{[\s\S]*?"
                r"bool do_reset = it == slot\.prompt\.checkpoints\.rend\(\);"
            ),
            rationale="checkpoint search predicate must require exact prefix containment for hybrid/recurrent state instead of the SWA range rule",
            mode="replace",
            text=(
                "if (pos_min >= pos_min_thold) {\n"
                "                                    // whether the checkpoints hold a state that is valid only at its exact\n"
                "                                    // final position (hybrid/recurrent memory), as opposed to a range (SWA)\n"
                "                                    const bool ckpt_exact = ctx_tgt_state_exact();\n"
                "\n"
                "                                    // search for a context checkpoint\n"
                "                                    const auto it = std::find_if(\n"
                "                                        slot.prompt.checkpoints.rbegin(),\n"
                "                                        slot.prompt.checkpoints.rend(),\n"
                "                                        [&](const auto & cur) {\n"
                "                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]\n"
                "                                            SLT_TRC(slot, \"checking checkpoint with [%d, %d] against %d...\\n\", cur.pos_min, cur.pos_max, pos_min_thold);\n"
                "                                            if (ckpt_exact) {\n"
                "                                                // usable only if the tokens up to and including its position are\n"
                "                                                // a prefix of the new prompt, with at least one token left to\n"
                "                                                // process [TAG_PROMPT_LOGITS]. the state is self-contained, so\n"
                "                                                // the SWA slack in pos_min_thold does not apply\n"
                "                                                return cur.pos_max < pos_next - (has_new_tokens ? 0 : 1);\n"
                "                                            }\n"
                "                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]\n"
                "                                            if (cur.pos_max > pos_next) {\n"
                "                                                return false;\n"
                "                                            }\n"
                "                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;\n"
                "                                        }\n"
                "                                    );\n"
                "\n"
                "                                    bool do_reset = it == slot.prompt.checkpoints.rend();"
            ),
            guard="ckpt_exact = ctx_tgt_state_exact",
            max_span_lines=25,
        ),
        Edit(
            id="checkpoint-restore-reset-log-level",
            anchor=(
                r"                                        n_past   = std::min\(slot\.prompt\.tokens\.size_up_to_pos\(pos_next\), \(size_t\) it->n_tokens\);\n"
                r"                                        SLT_TRC\(slot,"
            ),
            rationale="promote the restore/reset logs from TRC to INF (matches upstream's other checkpoint log promotions)",
            mode="replace",
            text=(
                "                                        SLT_INF(slot,"
            ),
            guard=r"SLT_INF\(slot, \"restored context checkpoint",
        ),
        Edit(
            id="checkpoint-erase-diverged-content",
            anchor=(
                r"                                for \(auto it = slot\.prompt\.checkpoints\.begin\(\); it != slot\.prompt\.checkpoints\.end\(\);\) \{\n"
                r"                                    const auto & cur = \*it;\n"
                r"                                    if \(cur\.pos_max > pos_next\) \{\n"
                r"                                        SLT_TRC\(slot,"
            ),
            rationale="erase checkpoints that cover diverged content, not just checkpoints past pos_next -- once new tokens decode, staleness becomes undetectable",
            mode="replace",
            text=(
                "                                // erase any checkpoints that cover diverged content - once the new\n"
                "                                // tokens are decoded, their staleness would become undetectable\n"
                "                                const llama_pos pos_stale = std::min(pos_next_lcp, slot.task->tokens.pos_next());\n"
                "\n"
                "                                // an exact checkpoint at the divergence position irreversibly contains\n"
                "                                // the diverged token, while a range (SWA) checkpoint gets that entry\n"
                "                                // overwritten when decoding resumes from it\n"
                "                                const llama_pos pos_stale_min = ctx_tgt_state_exact() ? pos_stale : pos_stale + 1;\n"
                "\n"
                "                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {\n"
                "                                    const auto & cur = *it;\n"
                "                                    if (cur.pos_max > pos_next || cur.pos_max >= pos_stale_min) {\n"
                "                                        SLT_INF(slot,"
            ),
            guard=r"const llama_pos pos_stale_min = ctx_tgt_state_exact\(\)",
        ),
    ),
)

SERVER_TASK_PATCH = FilePatch(
    path="tools/server/server-task.cpp",
    description="Fix prompt-cache entry selection: salvage entries with a "
                "usable checkpoint even below the 25% keep-ratio floor, and "
                "pick whichever cached entry saves the most work instead of "
                "requiring it to also beat the current slot's own f_keep "
                "(upstream PR, squashed with its own follow-up fix for the "
                "no-checkpoints and fully-contained cases)",
    edits=(
        Edit(
            id="server-prompt-cache-n-reusable",
            anchor=r"server_prompt_cache_state \* server_prompt_cache::alloc\(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft\) \{",
            rationale="add the helper computing how many tokens of a cached prompt are actually salvageable given its checkpoints",
            mode="insert_before",
            text=(
                "// how many tokens of a cached prompt could actually be reused if we restored it\n"
                "static int64_t server_prompt_cache_n_reusable(const server_prompt_cache_state & state, int lcp) {\n"
                "    // the cached prompt is fully contained in the new one - nothing has to be\n"
                "    // rolled back, so all of it is reusable\n"
                "    if (lcp == (int) state.prompt.tokens.size()) {\n"
                "        return lcp;\n"
                "    }\n"
                "\n"
                "    // no checkpoints - the memory can be rolled back to an arbitrary position,\n"
                "    // so the entire common prefix is reusable\n"
                "    if (state.prompt.checkpoints.empty()) {\n"
                "        return lcp;\n"
                "    }\n"
                "\n"
                "    // without a rollback-capable memory module (hybrid/recurrent), the restored\n"
                "    // state is usable only up to its newest context checkpoint at or below the\n"
                "    // divergence point - that checkpoint is what the caller falls back to\n"
                "    // instead of reprocessing the prompt from scratch\n"
                "    int64_t res = 0;\n"
                "\n"
                "    for (const auto & ckpt : state.prompt.checkpoints) {\n"
                "        if (ckpt.n_tokens <= lcp) {\n"
                "            res = std::max(res, ckpt.n_tokens);\n"
                "        }\n"
                "    }\n"
                "\n"
                "    return res;\n"
                "}\n"
                "\n"
            ),
            guard="server_prompt_cache_n_reusable",
        ),
        Edit(
            id="server-prompt-cache-load-n-reuse-min",
            anchor=(
                r"bool server_prompt_cache::load\(server_prompt & prompt, const server_tokens & tokens_new, "
                r"llama_context \* ctx_tgt, llama_context \* ctx_dft, int32_t id_slot\) \{\n"
                r"    const int lcp_best = prompt\.tokens\.get_common_prefix\(tokens_new\);"
            ),
            rationale="add the minimum-salvage floor used by the low-f_keep checkpoint rescue below",
            mode="replace",
            text=(
                "bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, "
                "llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {\n"
                "    // minimum salvage before we accept consuming a cached entry despite a low\n"
                "    // f_keep - restoring costs a state copy, so a tiny salvage is not worth it\n"
                "    constexpr int64_t n_reuse_min = 4096;\n"
                "\n"
                "    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);"
            ),
            guard="constexpr int64_t n_reuse_min = 4096;",
        ),
        Edit(
            id="server-prompt-cache-load-selection-rule",
            anchor=(
                r"        if \(f_keep_cur < 0\.25f\) \{\n"
                r"            continue;\n"
                r"        \}\n\n"
                r"        if \(f_keep_best < f_keep_cur && f_sim_best < f_sim_cur\) \{"
            ),
            rationale="salvage low-f_keep entries with a usable checkpoint, and select by f_sim (work saved) alone instead of also requiring the candidate to beat the current slot's own f_keep",
            mode="replace",
            text=(
                "        // don't consume a large cached prompt for a marginal salvage: restoring an\n"
                "        // entry removes it from the cache (states.erase() below).\n"
                "        //\n"
                "        // an entry that still carries a context checkpoint at or below the\n"
                "        // divergence point is worth taking even at a low f_keep - for\n"
                "        // hybrid/recurrent memory that checkpoint is the difference between\n"
                "        // resuming from the prefix and reprocessing the whole prompt\n"
                "        // ref: https://github.com/ggml-org/llama.cpp/issues/22746\n"
                "        if (f_keep_cur < 0.25f &&\n"
                "                server_prompt_cache_n_reusable(*it, lcp_cur) < n_reuse_min) {\n"
                "            continue;\n"
                "        }\n"
                "\n"
                "        // prefer whichever entry covers most of the new prompt, i.e. saves the\n"
                "        // most work. the slot's own prompt was already stored by prompt_save()\n"
                "        // before we got here, so picking a candidate can never lose it -\n"
                "        // requiring the candidate to also beat the slot's f_keep would reject\n"
                "        // useful entries whenever the slot happens to hold a small unrelated\n"
                "        // prompt, whose f_keep is then trivially high\n"
                "        if (f_sim_best < f_sim_cur) {"
            ),
            guard=r"server_prompt_cache_n_reusable\(\*it, lcp_cur\) < n_reuse_min",
        ),
    ),
)

PATCHES = [SERVER_CONTEXT_PATCH, SERVER_TASK_PATCH]
