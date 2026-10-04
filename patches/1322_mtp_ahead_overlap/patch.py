"""1322 (FMTP03/04): draft the next MTP front on the draft GPU while the target verifies, and promote it.

Today each speculative round is serial on the host: fresh MTP draft (6.4 ms at ~10K, 8.7 ms at ~80K, FMTP01), then
the target verify, whose sync wait (25.5 / 31.4 ms) leaves the 6900 idle. With BIGCHERRY_MTP_AHEAD=1:

1. Ahead draft (inside decode(), between the target llama_process() submit and llama_synchronize()): for a
   generating greedy slot whose verify batch carries a draft, the MTP drafter replays that draft as a forced front
   from the authoritative seed (1321 dp.forced) and continues it into a tail (dp.n_tail = n_draft_max + 1): the
   tail is the MTP prediction of the target's bonus token followed by the next front. The draft KV written beyond
   the checkpoint is removed again before common_speculative_process() reseeds it (same trim as after a normal
   draft), so the authoritative path is unchanged. Only for draft contexts with partial seq_rm (MTP head KV).
2. Promotion (accept): when the whole front was accepted and the target sampled the tail's first token, the rest of
   the (full-length) tail becomes the next round's draft and the serial fresh draft is skipped for that round (the
   round still checkpoints and trims as a fresh draft does). The promoted tokens are MTP chain depths k+2.. built on
   draft hidden rows, not a fresh draft reseeded from the target's hidden row (which only exists after verify), so
   they are accepted somewhat less than a fresh front; the gain is the skipped serial draft.

A promoted front is a proposal like any other draft: the target verifies it, so greedy output is unchanged. Any
ahead failure leaves the round on the ordinary path. A log line every 64 rounds reports rounds / ahead drafts /
promotions and the ahead host time (BIGCHERRY_MTP_AHEAD stats). Requires 1321.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_A_HELPER = "struct server_slot {\n"
_N_HELPER = r"""// bigcherry 1322 (FMTP03/04): MTP ahead draft during target verify + promotion (BIGCHERRY_MTP_AHEAD=1)
static bool bc_mtp_ahead_on() {
    static const bool on = getenv("BIGCHERRY_MTP_AHEAD") != nullptr && atoi(getenv("BIGCHERRY_MTP_AHEAD")) != 0;
    return on;
}

struct bc_mtp_ahead_stats {
    int64_t rounds = 0, ahead = 0, ahead_tokens = 0, promoted = 0, promoted_tokens = 0, ahead_us = 0;
};

static bc_mtp_ahead_stats & bc_mtp_ahead_st() {
    static bc_mtp_ahead_stats st;
    return st;
}

struct server_slot {
"""

_A_MEMBERS = ("    common_prompt_checkpoint spec_ckpt;\n"
              "    bool spec_is_replay = false;\n")
_N_MEMBERS = _A_MEMBERS + (
    "    // bigcherry 1322: front verified this round, MTP tail drafted ahead of it, and the tail rest promoted to the\n"
    "    // next round's draft\n"
    "    llama_tokens bc_ahead_front;\n"
    "    llama_tokens bc_ahead_tail;\n"
    "    llama_tokens bc_ahead_scratch;\n"
    "    llama_tokens bc_promoted;\n")

_A_RESET = ("            spec_draft.clear();\n"
            "            spec_i_batch.clear();\n"
            "            spec_ckpt.clear();\n")
_N_RESET = _A_RESET + (
    "            bc_ahead_front.clear();  // bigcherry 1322\n"
    "            bc_ahead_tail.clear();\n"
    "            bc_promoted.clear();\n")

_A_PROMOTE = ("                            /* .seed     = */ slot.task->params.sampling.seed,\n"
              "                        };\n"
              "\n"
              "                        drafting.push_back(&slot);\n")
_N_PROMOTE = ("                            /* .seed     = */ slot.task->params.sampling.seed,\n"
              "                        };\n"
              "\n"
              "                        // bigcherry 1322: a promoted ahead tail is this round's draft - skip the serial draft;\n"
              "                        // the slot still goes through the checkpoint/trim of a fresh draft below\n"
              "                        if (!slot.bc_promoted.empty() && !spec_reject) {\n"
              "                            const size_t n = std::min<size_t>(slot.bc_promoted.size(), (size_t) n_draft_max);\n"
              "                            slot.spec_draft.assign(slot.bc_promoted.begin(), slot.bc_promoted.begin() + n);\n"
              "                            common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;\n"
              "                            bc_mtp_ahead_st().promoted++;\n"
              "                            bc_mtp_ahead_st().promoted_tokens += (int64_t) n;\n"
              "                        }\n"
              "                        slot.bc_promoted.clear();\n"
              "\n"
              "                        drafting.push_back(&slot);\n")

_A_DECODE = "            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());\n"
_N_DECODE = _A_DECODE + r"""            // bigcherry 1322: draft ahead on the draft GPU while the target verifies (submit returned, sync not yet)
            if (ret == 0 && spec && ctx_dft && bc_mtp_ahead_on() && ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART) {
                for (auto & slot : slots) {
                    slot.bc_ahead_tail.clear();
                    if (slot.state != SLOT_STATE_GENERATING || slot.spec_draft.empty() || slot.spec_is_replay ||
                            slot.spec_i_batch.empty() || slot.use_spec_rejection()) {
                        continue;
                    }
                    const int64_t bc_t0 = ggml_time_us();
                    auto & dp = common_speculative_get_draft_params(spec.get(), slot.id);
                    const common_speculative_draft_params saved = dp;
                    slot.bc_ahead_front = slot.spec_draft;
                    slot.bc_ahead_scratch.clear();
                    dp.drafting = true;
                    dp.result   = &slot.bc_ahead_scratch;
                    dp.result_q = nullptr;
                    dp.forced   = &slot.bc_ahead_front;
                    // bonus-token prediction + a full next front, independent of this round's (possibly short) front
                    dp.n_tail   = (int32_t) slot.get_n_draft_max() + 1;
                    dp.tail     = &slot.bc_ahead_tail;
                    common_speculative_draft(spec.get());
                    dp = saved;
                    dp.drafting = false;
                    // the ahead decode wrote draft KV past the checkpoint: trim it as after a normal draft, so
                    // common_speculative_process() reseeds from the same state
                    if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, slot.spec_ckpt.pos_max + 1, -1)) {
                        GGML_ABORT("bigcherry 1322: failed to trim the ahead draft of sequence %d\n", slot.id);
                    }
                    if (slot.bc_ahead_tail.size() < 2) {  // bonus prediction + at least one next-front token
                        slot.bc_ahead_tail.clear();
                    }
                    bc_mtp_ahead_st().ahead++;
                    bc_mtp_ahead_st().ahead_tokens += (int64_t) slot.bc_ahead_tail.size();
                    bc_mtp_ahead_st().ahead_us += ggml_time_us() - bc_t0;
                }
            }
"""

_A_ACCEPT = ("                slot.spec_draft = std::move(accepted);\n"
             "            }\n"
             "\n"
             "            const auto ids = std::move(slot.spec_draft);\n")
_N_ACCEPT = ("                // bigcherry 1322: whole front accepted and the target sampled the tail's bonus prediction -> the\n"
             "                // rest of the tail continues this state (on draft hidden rows); promote it to the next round's draft\n"
             "                slot.bc_promoted.clear();\n"
             "                if (bc_mtp_ahead_on()) {\n"
             "                    // only a full-length tail (bonus + n_draft_max) may replace a fresh draft: a short promoted\n"
             "                    // front shrinks the verify batch and loses to a fresh full draft (RV4217 / deep-dive 2)\n"
             "                    if (slot.bc_ahead_tail.size() == (size_t) slot.get_n_draft_max() + 1 &&\n"
             "                            accepted.size() == slot.spec_draft.size() + 1 &&\n"
             "                            slot.spec_draft == slot.bc_ahead_front && accepted.back() == slot.bc_ahead_tail[0]) {\n"
             "                        slot.bc_promoted.assign(slot.bc_ahead_tail.begin() + 1, slot.bc_ahead_tail.end());\n"
             "                    }\n"
             "                    slot.bc_ahead_tail.clear();\n"
             "                    bc_mtp_ahead_stats & st = bc_mtp_ahead_st();\n"
             "                    if (++st.rounds % 64 == 0) {\n"
             "                        SLT_WRN(slot, \"BIGCHERRY_MTP_AHEAD rounds=%lld ahead=%lld ahead_tokens=%lld promoted=%lld promoted_tokens=%lld ahead_us=%lld\\n\",\n"
             "                            (long long) st.rounds, (long long) st.ahead, (long long) st.ahead_tokens, (long long) st.promoted,\n"
             "                            (long long) st.promoted_tokens, (long long) st.ahead_us);\n"
             "                    }\n"
             "                }\n"
             "\n"
             + _A_ACCEPT)

PATCHES = [
    FilePatch(
        path="tools/server/server-context.cpp",
        description="1322: MTP ahead draft during target verify + promotion (BIGCHERRY_MTP_AHEAD=1)",
        language="none",
        edits=(
            Edit(id="mtp-ahead-helper", anchor=re.escape(_A_HELPER), mode="replace", text=_N_HELPER,
                 guard=r"bigcherry 1322 \(FMTP03/04\): MTP ahead draft", rationale="Before struct server_slot.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="mtp-ahead-members", anchor=re.escape(_A_MEMBERS), mode="replace", text=_N_MEMBERS,
                 guard=r"llama_tokens bc_ahead_front;", rationale="server_slot speculative members.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="mtp-ahead-reset", anchor=re.escape(_A_RESET), mode="replace", text=_N_RESET,
                 guard=r"bc_ahead_front\.clear\(\);  // bigcherry 1322", rationale="server_slot reset of speculative state.",
                 expect_matches=1, max_span_lines=4),
            Edit(id="mtp-ahead-promote", anchor=re.escape(_A_PROMOTE), mode="replace", text=_N_PROMOTE,
                 guard=r"bigcherry 1322: a promoted ahead tail is this round's draft", rationale="Fresh-draft setup in update_slots.",
                 expect_matches=1, max_span_lines=5),
            Edit(id="mtp-ahead-decode", anchor=re.escape(_A_DECODE), mode="replace", text=_N_DECODE,
                 guard=r"bigcherry 1322: draft ahead on the draft GPU", rationale="decode(): target submit before sync.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="mtp-ahead-accept", anchor=re.escape(_A_ACCEPT), mode="replace", text=_N_ACCEPT,
                 guard=r"bigcherry 1322: whole front accepted", rationale="Full-acceptance path after common_speculative_accept.",
                 expect_matches=1, max_span_lines=5),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_MTP_AHEAD", "0|1", "0",
           "experimental: draft the next MTP front on the draft GPU during target verify and promote it on full acceptance"),
)
