"""PRBE52: wire the existing NRO06 adaptive MTP depth controller into runtime MTP."""

import re

from bigcherry.patcher import Edit, FilePatch

_COMMON_H_INSERT = "    int32_t n_min_adaptive = 0; // adaptive MTP floor; 0 disables adaptive depth\n"

_ARG_OLD = """    add_opt(common_arg(
        {\"--spec-draft-n-min\"}, \"N\",
        string_format(\"minimum number of draft tokens to use for speculative decoding (default: %d)\", params.speculative.draft.n_min),
        [](common_params & params, int value) {
            params.speculative.draft.n_min = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env(\"LLAMA_ARG_SPEC_DRAFT_N_MIN\"));
"""
_ARG_NEW = _ARG_OLD + """    add_opt(common_arg(
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

_INCLUDES_OLD = """#include <algorithm>
#include <cassert>
"""
_INCLUDES_NEW = """#include <algorithm>
#include <atomic>
#include <cassert>
"""

_MEMBERS_OLD = """    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
"""
_MEMBERS_NEW = """    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    // PRBE52: per-sequence adaptive depth state. n_min_adaptive==0 is disabled.
    std::vector<bigcherry_nro06_adaptive_mtp> adaptive_state;
    std::vector<int32_t>                      last_n_draft;

    std::vector<int32_t> i_batch_beg;
"""

_CTOR_OLD = """        }
        this->n_max = this->params.n_max;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_last.assign(n_seq, -1);
"""
_CTOR_NEW = """        }
        if (this->params.n_min_adaptive < 0 || this->params.n_min_adaptive > this->params.n_max) {
            throw std::invalid_argument(\"adaptive MTP floor must be 0 (disabled) or <= --spec-draft-n-max\");
        }
        this->n_max = this->params.n_max;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
        adaptive_state.resize(n_seq);
        last_n_draft.assign(n_seq, 0);

        i_last.assign(n_seq, -1);
"""

_BEGIN_OLD = """    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
"""
_BEGIN_NEW = """    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (params.n_min_adaptive > 0) {
            adaptive_state.at(seq_id).reset(params.n_max, params.n_min_adaptive);
            last_n_draft.at(seq_id) = 0;
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
"""

_DRAFT_START_OLD = """            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());
"""
_DRAFT_START_NEW = """            n_drafting++;
            drafting[seq_id] = true;
            last_n_draft[seq_id] = 0;
            common_sampler_reset(smpls[seq_id].get());
"""

_LIMIT_OLD = """                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }
"""
_LIMIT_NEW = """                result.push_back(id);

                const int effective_n_max = params.n_min_adaptive > 0 ? adaptive_state[seq_id].n_cur : params.n_max;
                if (params.n_min_adaptive > 0 && getenv(\"BIGCHERRY_PATCH_TRACE\") != nullptr) {
                    static std::atomic_flag bigcherry_prbe52_logged = ATOMIC_FLAG_INIT;
                    if (!bigcherry_prbe52_logged.test_and_set(std::memory_order_relaxed)) {
                        SPC_WRN(\"BIGCHERRY_PATCH_HIT patch=1268_prbe52_adaptive_mtp_wiring path=mtp_adaptive_depth contract=PRBE52-ADAPTIVE-MTP-WIRING depth=%d seq=%d\\n\",
                                effective_n_max, (int) seq_id);
                    }
                }
                if (effective_n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }
"""

_FINALIZE_OLD = """            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
"""
_FINALIZE_NEW = """            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
            last_n_draft[seq_id] = (int32_t) dp.result->size();
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
"""

_ACCEPT_OLD = """        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }
};
"""
_ACCEPT_NEW = """        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);

        if (params.n_min_adaptive > 0) {
            adaptive_state[seq_id].update(last_n_draft[seq_id], (int) n_accepted, params.n_max, params.n_min_adaptive);
        }
    }
};
"""

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
        edits=(Edit(id="prbe52-adaptive-arg", anchor=re.escape(_ARG_OLD), mode="replace", text=_ARG_NEW,
                    guard=re.escape("LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE"), rationale="Opt in explicitly; negative values fail parsing.", expect_matches=1, max_span_lines=8),),
    ),
    FilePatch(
        path="common/speculative.cpp",
        description="PRBE52 per-sequence adaptive MTP runtime wiring",
        edits=(
            Edit(id="prbe52-atomic-include", anchor=re.escape(_INCLUDES_OLD), mode="replace", text=_INCLUDES_NEW,
                 guard=r"#include <atomic>", rationale="Thread-safe once-per-process activation evidence.", expect_matches=1, max_span_lines=2),
            Edit(id="prbe52-state", anchor=re.escape(_MEMBERS_OLD), mode="replace", text=_MEMBERS_NEW,
                 guard=r"adaptive_state", rationale="Keep adaptive state independent per sequence.", expect_matches=1, max_span_lines=3),
            Edit(id="prbe52-ctor", anchor=re.escape(_CTOR_OLD), mode="replace", text=_CTOR_NEW,
                 guard=re.escape("adaptive MTP floor must be 0"), rationale="Validate the post-chain-head cap and size per-sequence state.", expect_matches=1, max_span_lines=7),
            Edit(id="prbe52-begin-reset", anchor=re.escape(_BEGIN_OLD), mode="replace", text=_BEGIN_NEW,
                 guard=r"adaptive_state\.at\(seq_id\)\.reset", rationale="Reset adaptation at the request/sequence begin boundary.", expect_matches=1, max_span_lines=6),
            Edit(id="prbe52-draft-reset", anchor=re.escape(_DRAFT_START_OLD), mode="replace", text=_DRAFT_START_NEW,
                 guard=r"last_n_draft\[seq_id\] = 0", rationale="Prevent stale draft accounting on zero-draft exits.", expect_matches=1, max_span_lines=3),
            Edit(id="prbe52-depth-limit", anchor=re.escape(_LIMIT_OLD), mode="replace", text=_LIMIT_NEW,
                 guard=re.escape("PRBE52-ADAPTIVE-MTP-WIRING"), rationale="Use adaptive depth only when explicitly enabled; fixed-depth code remains the default.", expect_matches=1, max_span_lines=7),
            Edit(id="prbe52-draft-accounting", anchor=re.escape(_FINALIZE_OLD), mode="replace", text=_FINALIZE_NEW,
                 guard=r"last_n_draft\[seq_id\] = \(int32_t\) dp\.result->size\(\)", rationale="Update only from drafts that survive the existing n_min filter.", expect_matches=1, max_span_lines=8),
            Edit(id="prbe52-accept-update", anchor=re.escape(_ACCEPT_OLD), mode="replace", text=_ACCEPT_NEW,
                 guard=r"adaptive_state\[seq_id\]\.update", rationale="Feed the real accepted count back to the existing controller.", expect_matches=1, max_span_lines=6),
        ),
    ),
]
