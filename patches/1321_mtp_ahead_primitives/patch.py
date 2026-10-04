"""1321 (FMTP02): MTP draft primitives for the ahead pipeline - forced promoted front + live tail continuation.

FMTP needs two things from the single-head MTP drafter (qwen35 / qwen4exp), both without a second MTP loop:

1. Forced-front replay. A promoted front (tokens already drafted ahead and now being target-verified) is fed through
   the MTP head from the authoritative seed (id_last, pending_h at pos0) as forced inputs: each forced token is paired
   with the hidden row the preceding step produced, exactly as a sampled token would be. p_min does not apply to
   forced tokens (they are already chosen). After the last forced token, drafting continues by sampling.
2. Live continuation. With n_tail > 0, drafting continues past the normal front (params.n_max) for up to n_tail more
   tokens, which go to dp.tail - disjoint from dp.result, so the verify set is unchanged. Because the tail is drafted
   in the same draft() call, from the live ctx_dft KV / sampler frontier, no continuation lease has to outlive a
   rollback or reseed (FMTP02 review finding): the next round reseeds as before.

New common_speculative_draft_params fields (defaults keep today's behaviour exactly):
    const llama_tokens * forced = nullptr;  // promoted front, forced in order (greedy drafting only)
    int32_t              n_tail = 0;        // extra tokens to draft past n_max into *tail
    llama_tokens *       tail   = nullptr;  // output; cleared at draft() start when n_tail > 0
Only the single-head non-shared MTP path supports them (asserted); chained heads and shared-KV MTP do not.
No scheduling change: the server does not set the fields yet (FMTP03).
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_A_H = ("    // the target's temp and seed, read only when the drafter samples probabilistically\n"
        "    float    temp = 1.0f;\n"
        "    uint32_t seed = LLAMA_DEFAULT_SEED;\n")
# appended after the last field: the server brace-initialises the leading fields positionally
_N_H = _A_H + (
    "\n"
    "    // bigcherry 1321 (FMTP02), single-head MTP only: a promoted front forced through the MTP head in order before\n"
    "    // sampling resumes (greedy drafting only; p_min does not apply to forced tokens), and a live tail drafted past\n"
    "    // the front into *tail (disjoint from *result). Defaults keep the ordinary draft.\n"
    "    const llama_tokens * forced = nullptr;\n"
    "    int32_t              n_tail = 0;\n"
    "    llama_tokens *       tail   = nullptr;\n"
    "    float                tail_p_min = 0.0f;  // stricter confidence cut for tail tokens (max with p_min)\n")

_A_SEED = ("            const int32_t idx = batch.add(dp.id_last, dp.pos0, seq_id, true);\n"
           "            batch.set_embd(idx, { pending_h[seq_id].data(), 1, (size_t) n_embd });\n")
_N_SEED = ("            // bigcherry 1321: forced front / live tail are single-head non-shared MTP only, greedy only\n"
           "            GGML_ASSERT(((dp.forced == nullptr || dp.forced->empty()) && dp.n_tail <= 0) || (!chain_heads && !is_mem_shared));\n"
           "            GGML_ASSERT(dp.forced == nullptr || dp.forced->empty() || dp.result_q == nullptr);\n"
           "            GGML_ASSERT(dp.n_tail <= 0 || dp.tail != nullptr);\n"
           "            GGML_ASSERT(dp.forced == nullptr || (int) dp.forced->size() <= params.n_max);\n"
           "            if (dp.n_tail > 0) {\n"
           "                dp.tail->clear();\n"
           "            }\n"
           + _A_SEED)

_A_STEP = """                // add drafted token for each sequence
                const llama_token id = dparams.at(seq_id).result_q ? id_sampled : cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (dp.result_q) {
                    dp.result_q->emplace_back(cur_p->data, cur_p->data + cur_p->size);
                }

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }
"""
_N_STEP = """                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                // bigcherry 1321: forced promoted front first, then sampled tokens; past the front (the forced tokens
                // when given, else n_max) they go to the tail
                const size_t bc_n_forced = dp.forced ? dp.forced->size() : 0;
                const size_t bc_front    = dp.forced ? bc_n_forced : (size_t) params.n_max;
                const bool   bc_forced   = result.size() < bc_n_forced;
                const bool   bc_in_tail  = !bc_forced && bc_front <= result.size();

                // add drafted token for each sequence
                const llama_token id = bc_forced ? (*dp.forced)[result.size()]
                                     : dp.result_q ? id_sampled : cur_p->data[0].id;

                // only collect very high-confidence draft tokens (a forced token is already chosen; tail tokens
                // may use a stricter cut)
                const float bc_p_min = bc_in_tail ? std::max(params.p_min, dp.tail_p_min) : params.p_min;
                if (!bc_forced && cur_p->data[0].p < bc_p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                if (bc_in_tail) {
                    dp.tail->push_back(id);
                    if (dp.n_tail <= (int) dp.tail->size()) {
                        drafting[seq_id] = false;
                        n_drafting--;
                        continue;
                    }
                } else {
                    result.push_back(id);

                    if (dp.result_q) {
                        dp.result_q->emplace_back(cur_p->data, cur_p->data + cur_p->size);
                    }

                    if (bc_front <= result.size() && dp.n_tail <= 0) {
                        drafting[seq_id] = false;
                        n_drafting--;
                        continue;
                    }
                }
"""

# the n_min post-pass also exists in the draft-simple drafters: anchor on the MTP-only chained-head restore before it
_NMIN_CTX = ("            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes\n"
             "        }\n"
             "\n"
             "        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {\n"
             "            auto & dp = dparams[seq_id];\n"
             "            if (!dp.drafting) {\n"
             "                continue;\n"
             "            }\n"
             "\n")
_A_NMIN = _NMIN_CTX + ("            if (dp.result->size() < (size_t) params.n_min) {\n"
                       "                dp.result->clear();\n"
                       "            }\n")
_N_NMIN = _NMIN_CTX + ("            if (dp.result->size() < (size_t) params.n_min) {\n"
           "                dp.result->clear();\n"
           "                if (dp.n_tail > 0) {  // bigcherry 1321: a tail only continues a front that is verified\n"
           "                    dp.tail->clear();\n"
           "                }\n"
           "            }\n")

PATCHES = [
    FilePatch(
        path="common/speculative.h",
        description="1321: forced-front / live-tail fields in common_speculative_draft_params",
        language="none",
        edits=(
            Edit(id="mtp-ahead-params", anchor=re.escape(_A_H), mode="replace", text=_N_H,
                 guard=r"bigcherry 1321 \(FMTP02\), single-head MTP only", rationale="After the last draft-params field (seed).",
                 expect_matches=1, max_span_lines=4),
        ),
    ),
    FilePatch(
        path="common/speculative.cpp",
        description="1321: MTP draft loop forced promoted front + live tail continuation",
        language="none",
        edits=(
            Edit(id="mtp-ahead-seed", anchor=re.escape(_A_SEED), mode="replace", text=_N_SEED,
                 guard=r"bigcherry 1321: forced front / live tail are single-head", rationale="MTP draft() seed per sequence.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="mtp-ahead-step", anchor=re.escape(_A_STEP), mode="replace", text=_N_STEP,
                 guard=r"bigcherry 1321: forced promoted front first", rationale="MTP draft() per-step token selection.",
                 expect_matches=1, max_span_lines=28),
            Edit(id="mtp-ahead-nmin", anchor=re.escape(_A_NMIN), mode="replace", text=_N_NMIN,
                 guard=r"bigcherry 1321: a tail only continues", rationale="MTP draft() n_min post-pass.",
                 expect_matches=1, max_span_lines=13),
        ),
    ),
]
