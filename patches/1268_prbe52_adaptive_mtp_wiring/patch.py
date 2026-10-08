"""PRBE52: wire the existing NRO06 adaptive MTP depth controller into runtime MTP."""

from bigcherry.patcher import Edit, FilePatch

_COMMON_H_INSERT = "    int32_t n_min_adaptive = 0; // adaptive MTP floor; 0 disables adaptive depth\n"

_ARG_NEW = """    add_opt(common_arg(
        {\"--spec-draft-n-min\"}, \"N\",
        string_format(\"minimum number of draft tokens to use for speculative decoding (default: %d)\", params.speculative.draft.n_min),
        [](common_params & params, int value) {
            params.speculative.draft.n_min = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env(\"LLAMA_ARG_SPEC_DRAFT_N_MIN\"));
    add_opt(common_arg(
        {\"--spec-draft-n-min-adaptive\"}, \"N\",
        string_format(\"adaptive MTP draft-depth floor; 0 disables (default: %d)\", params.speculative.draft.n_min_adaptive),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument(\"invalid value\");
            }
            params.speculative.draft.n_min_adaptive = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env(\"LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE\"));
"""

_INCLUDES_NEW = """#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
"""

_MEMBERS_NEW = """    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    // PRBE52: per-sequence adaptive depth state. n_min_adaptive==0 is disabled.
    std::vector<bigcherry_nro06_adaptive_mtp> adaptive_state;
    std::vector<int32_t>                      last_n_draft;

    std::vector<int32_t> i_batch_beg;
"""

_CTOR_NEW = """        }
        if (this->params.n_min_adaptive < 0 || this->params.n_min_adaptive > this->params.n_max) {
            throw std::invalid_argument(\"adaptive MTP floor must be 0 (disabled) or <= --spec-draft-n-max\");
        }
        if (this->params.n_min_adaptive > 0 && this->params.n_min_adaptive < this->params.n_min) {
            // GPT code review 2026-09-27 (req_6c90e1ebba83464a): an adaptive
            // floor BELOW n_min is a stuck state, not just a suboptimal one --
            // finalize() clears any draft shorter than n_min regardless of why
            // it stopped, so a controller capped at n_min_adaptive < n_min
            // would have every one of its drafts discarded forever, with the
            // acceptance signal that is supposed to grow n_cur back up never
            // arriving (dp.result was cleared, so accept() never even runs).
            throw std::invalid_argument(\"adaptive MTP floor must be 0 (disabled) or >= --spec-draft-n-min\");
        }
        if (this->params.n_min_adaptive > 0) {
            const char * mtp_ahead = std::getenv(\"BIGCHERRY_MTP_AHEAD\");
            if (mtp_ahead != nullptr && std::atoi(mtp_ahead) != 0) {
                throw std::invalid_argument(\"adaptive MTP depth cannot be combined with BIGCHERRY_MTP_AHEAD=1\");
            }
        }
        this->n_max = this->params.n_max;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
        adaptive_state.resize(n_seq);
        last_n_draft.assign(n_seq, 0);

        i_last.assign(n_seq, -1);
"""

_BEGIN_NEW = """
        if (params.n_min_adaptive > 0) {
            adaptive_state.at(seq_id).reset(params.n_max, params.n_min_adaptive);
            last_n_draft.at(seq_id) = 0;
        }
"""

_DRAFT_START_NEW = """            last_n_draft[seq_id] = 0;
            if (params.n_min_adaptive > 0 && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
                static std::atomic_flag bigcherry_prbe52_logged = ATOMIC_FLAG_INIT;
                if (!bigcherry_prbe52_logged.test_and_set(std::memory_order_relaxed)) {
                    SPC_WRN("BIGCHERRY_PATCH_HIT patch=1268_prbe52_adaptive_mtp_wiring path=mtp_adaptive_depth contract=PRBE52-ADAPTIVE-MTP-WIRING depth=%d seq=%d\\n",
                            adaptive_state[seq_id].n_cur, (int) seq_id);
                }
            }
"""

_LIMIT_NEW = """                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                // PRBE52: keep the native/static n_max stop block byte-for-byte so 1321 can
                // replace it with forced-front/live-tail logic. The adaptive cap remains
                // immediately after that seam and caps the composed fresh front.
                if (params.n_min_adaptive > 0 &&
                        adaptive_state[seq_id].n_cur <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }
"""

_FINALIZE_NEW = """            last_n_draft[seq_id] = (int32_t) dp.result->size();
"""

_ACCEPT_NEW = """        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);

        if (params.n_min_adaptive > 0) {
            const int old_depth = adaptive_state[seq_id].n_cur;
            adaptive_state[seq_id].update(last_n_draft[seq_id], (int) n_accepted, params.n_max, params.n_min_adaptive);
            const int new_depth = adaptive_state[seq_id].n_cur;
            if (new_depth != old_depth && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
                SPC_WRN("BIGCHERRY_PATCH_HIT patch=1268_prbe52_adaptive_mtp_wiring path=mtp_adaptive_depth contract=PRBE52-ADAPTIVE-MTP-WIRING event=depth_change old=%d new=%d drafted=%d accepted=%d seq=%d\\n",
                        old_depth, new_depth, last_n_draft[seq_id], (int) n_accepted, (int) seq_id);
            }
        }
    }
};
"""

# Anchors intentionally contain code tokens only. apply.py blanks comments and
# C/C++ string literals at the same offsets before regex matching.
_ARG_ANCHOR = (
    r"    add_opt\(common_arg\(\n"
    r"        [^\n]*\n"
    r"        string_format\([^\n]*params\.speculative\.draft\.n_min\),\n"
    r"        \[\]\(common_params & params, int value\) \{\n"
    r"            params\.speculative\.draft\.n_min = value;\n"
    r"        \}\n"
    r"    \)\.set_spec\(\)\.set_examples\(\{LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI\}\)\.set_env\([^\n]*\);\n"
)
_INCLUDES_ANCHOR = r"#include <algorithm>\n#include <cassert>\n"
_MEMBERS_ANCHOR = r"    std::vector<std::vector<float>> pending_h;[^\n]*\n\n    std::vector<int32_t> i_batch_beg;\n"
_CTOR_ANCHOR = (
    r"        \}\n"
    r"        this->n_max = this->params.n_max;\n\n"
    r"        pending_h.assign\(n_seq, std::vector<float>\(n_embd, 0\.0f\)\);\n\n"
    r"        i_last.assign\(n_seq, -1\);\n"
)
# b11402 (upstream #27694, probabilistic MTP): begin() now resets the per-sequence sampler first, the draft start
# decides between greedy and rejection-sampling drafts before any sampler reset, and the depth cap follows the
# optional candidate capture. The three anchors below attach to that shape; each edit inserts its own lines only.
_BEGIN_ANCHOR = (
    r"    void begin\(llama_seq_id seq_id, const llama_tokens & prompt\) override \{\n"
    r"[^\n]*\n"
    r"        common_sampler_reset\(smpls\[seq_id\]\.get\(\)\);\n"
    # Other draft classes open begin() the same way; only the MTP class follows the empty-prompt return with the
    # shared-memory position check.
    r"(?=\n        const int32_t N = \(int32_t\) prompt.size\(\);\n"
    r"        if \(N <= 0\) \{\n"
    r"            return;\n"
    r"        \}\n"
    r"\n        auto \* ctx_dft = this->params\.ctx_dft;\n"
    r"        const llama_pos pos_max = [^\n]*\n\n"
    r"        if \(pos_max < N - 1 && !is_mem_shared\))"
)
_DRAFT_START_ANCHOR = (
    r"            n_drafting\+\+;\n"
    r"            drafting\[seq_id\] = true;\n"
    # The pair exists in several draft implementations. MTP alone follows it with the greedy / probabilistic
    # decision and later feeds pending_h into the embedding batch.
    r"(?=[^\n]*\n"
    r"            if \(!params\.probabilistic\) \{\n"
    r"                dp\.result_q = nullptr;\n"
    r"            \}\n"
    r"(?:[^\n]*\n){1,16}?"
    r"            batch\.set_embd\(idx, \{ pending_h\[seq_id\]\.data\(\), 1, \(size_t\) n_embd \}\);)"
)
_LIMIT_ANCHOR = (
    r"                if \(params.n_max <= \(int\) result.size\(\)\) \{\n"
    r"                    drafting\[seq_id\] = false;\n"
    r"                    n_drafting--;\n"
    r"                    continue;\n"
    r"                \}\n"
    r"(?=\n                if \(chain_heads\) \{)"
)

_FINALIZE_ANCHOR = (
    r"            if \(dp.result->size\(\) < \(size_t\) params.n_min\) \{\n"
    r"                dp.result->clear\(\);\n"
    r"            \}\n"
    r"(?=        \}\n"
    r"    \}\n\n"
    r"    void accept\(llama_seq_id seq_id, uint16_t n_accepted, bool[^\n]*\) override \{\n"
    r"        if \(seq_id < 0 \|\| seq_id >= \(llama_seq_id\) n_seq\) \{\n"
    r"            return;\n"
    r"        \}\n\n"
    r"        const int32_t n_rows = verify_h_rows\[seq_id\];)"
)

_ACCEPT_ANCHOR = (
    r"        const int32_t i_h = std::min<int32_t>\(n_accepted, n_rows - 1\);\n"
    r"        const size_t row_bytes = \(size_t\) n_embd \* sizeof\(float\);\n"
    r"        std::memcpy\(pending_h\[seq_id\]\.data\(\), verify_h\[seq_id\]\.data\(\) \+ \(size_t\) i_h \* n_embd, row_bytes\);\n"
    r"    \}\n"
    r"\};\n"
)

PATCHES = [
    FilePatch(
        path="common/common.h",
        description="PRBE52 explicit disabled-by-default adaptive MTP floor",
        edits=(Edit(id="prbe52-adaptive-param", anchor=r"    int32_t n_min = 0;[^\n]*\n", mode="insert_after", text=_COMMON_H_INSERT,
                    guard=r"n_min_adaptive = 0", rationale="Anchor only on the stable n_min field so lower-order composed fields cannot invalidate the edit.", expect_matches=1, max_span_lines=2),),
    ),
    FilePatch(
        path="common/arg.cpp",
        description="PRBE52 adaptive MTP CLI/env option",
        edits=(Edit(id="prbe52-adaptive-arg", anchor=_ARG_ANCHOR, mode="replace", text=_ARG_NEW,
                    guard=r"LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE", rationale="Identify the n_min option by executable code tokens; strings are noise-stripped.", expect_matches=1, max_span_lines=8),),
    ),
    FilePatch(
        path="common/speculative.cpp",
        description="PRBE52 per-sequence adaptive MTP runtime wiring",
        edits=(
            Edit(id="prbe52-atomic-include", anchor=_INCLUDES_ANCHOR, mode="replace", text=_INCLUDES_NEW,
                 guard=r"#include <atomic>", rationale="Thread-safe once-per-process activation evidence.", expect_matches=1, max_span_lines=3),
            Edit(id="prbe52-state", anchor=_MEMBERS_ANCHOR, mode="replace", text=_MEMBERS_NEW,
                 guard=r"adaptive_state", rationale="Identify member declarations by code only; trailing comments are noise-stripped.", expect_matches=1, max_span_lines=4),
            Edit(id="prbe52-ctor", anchor=_CTOR_ANCHOR, mode="replace", text=_CTOR_NEW,
                 guard=r"n_min_adaptive > this->params.n_max", rationale="Validate the post-chain-head cap and size per-sequence state.", expect_matches=1, max_span_lines=7),
            Edit(id="prbe52-begin-reset", anchor=_BEGIN_ANCHOR, mode="insert_after", text=_BEGIN_NEW,
                 guard=r"adaptive_state\.at\(seq_id\)\.reset", rationale="Reset adaptation at the request/sequence begin boundary.", expect_matches=1, max_span_lines=4),
            Edit(id="prbe52-draft-reset", anchor=_DRAFT_START_ANCHOR, mode="insert_after", text=_DRAFT_START_NEW,
                 guard=r"last_n_draft\[seq_id\] = 0", rationale="Identify the MTP drafting start by its pending-h embedding batch setup, not by a pair shared with other draft implementations.", expect_matches=1, max_span_lines=3),
            Edit(id="prbe52-depth-limit", anchor=_LIMIT_ANCHOR, mode="replace", text=_LIMIT_NEW,
                 guard=r"adaptive_state\[seq_id\]\.n_cur <= \(int\) result\.size\(\)", rationale="Preserve the native n_max stop block for 1321, then apply the adaptive cap before chain-head continuation.", expect_matches=1, max_span_lines=6),
            Edit(id="prbe52-draft-accounting", anchor=_FINALIZE_ANCHOR, mode="insert_after", text=_FINALIZE_NEW,
                 guard=r"last_n_draft\[seq_id\] = \(int32_t\) dp\.result->size\(\)", rationale="Insert accounting after the native n_min block, preserving 1321's exact n_min preimage.", expect_matches=1, max_span_lines=8),
            Edit(id="prbe52-accept-update", anchor=_ACCEPT_ANCHOR, mode="replace", text=_ACCEPT_NEW,
                 guard=r"adaptive_state\[seq_id\]\.update", rationale="Feed the real accepted count back; anchor only on executable statements.", expect_matches=1, max_span_lines=6),
        ),
    ),
]
