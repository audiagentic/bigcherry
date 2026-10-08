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

    // PRBE52: per-sequence adaptive state plus request-independent depth evidence.
    std::vector<bigcherry_nro06_adaptive_mtp> adaptive_state;
    std::vector<std::vector<uint64_t>>        adaptive_depth_hist;

    std::vector<int32_t> i_batch_beg;
"""

_CTOR_NEW = """        }
        if (this->params.n_min_adaptive < 0 || this->params.n_min_adaptive > this->params.n_max) {
            throw std::invalid_argument(\"adaptive MTP floor must be 0 (disabled) or <= --spec-draft-n-max\");
        }
        if (this->params.n_min_adaptive > 0 && this->params.n_min_adaptive < this->params.n_min) {
            throw std::invalid_argument(\"adaptive MTP floor must be 0 (disabled) or >= --spec-draft-n-min\");
        }
        this->n_max = this->params.n_max;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
        adaptive_state.resize(n_seq);
        adaptive_depth_hist.assign(n_seq, std::vector<uint64_t>((size_t) this->params.n_max + 1, 0));

        i_last.assign(n_seq, -1);
"""

_BEGIN_NEW = """
        if (params.n_min_adaptive > 0) {
            adaptive_state.at(seq_id).reset(params.n_max, params.n_min_adaptive);
        }
"""

_DRAFT_START_NEW = """            if (params.n_min_adaptive > 0 && dp.n_tail <= 0 && getenv(\"BIGCHERRY_PATCH_TRACE\") != nullptr) {
                static std::atomic_flag bigcherry_prbe52_logged = ATOMIC_FLAG_INIT;
                if (!bigcherry_prbe52_logged.test_and_set(std::memory_order_relaxed)) {
                    SPC_WRN(\"BIGCHERRY_PATCH_HIT patch=1268_prbe52_adaptive_mtp_wiring path=mtp_adaptive_depth contract=PRBE52-ADAPTIVE-MTP-WIRING depth=%d seq=%d\\n\",
                            adaptive_state[seq_id].n_cur, (int) seq_id);
                }
            }
"""

_LIMIT_NEW = """                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                // PRBE52: ordinary fresh fronts obey the deterministic per-sequence cap.
                // 1321 live-tail calls carry an adaptive-sized forced front plus an explicit
                // n_tail budget from the server, so they must be allowed to continue.
                const int bc_adaptive_cap = dp.n_max > 0
                    ? std::min(adaptive_state[seq_id].n_cur, dp.n_max)
                    : adaptive_state[seq_id].n_cur;
                if (params.n_min_adaptive > 0 && dp.n_tail <= 0 &&
                        bc_adaptive_cap <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }
"""

_FINALIZE_NEW = """
"""

_ACCEPT_NEW = """        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);

        if (params.n_min_adaptive > 0) {
            // verify_h_rows is authoritative for the front actually verified this round.
            // This covers 1322-promoted fronts, which bypass a fresh draft() call.
            const int n_draft_actual = std::max(0, n_rows - 1);
            if (n_draft_actual < (int) adaptive_depth_hist[seq_id].size()) {
                adaptive_depth_hist[seq_id][n_draft_actual]++;
            }

            const int old_depth = adaptive_state[seq_id].n_cur;
            adaptive_state[seq_id].update(n_draft_actual, (int) n_accepted, params.n_max, params.n_min_adaptive);
            const int new_depth = adaptive_state[seq_id].n_cur;
            if (new_depth != old_depth && getenv(\"BIGCHERRY_PATCH_TRACE\") != nullptr) {
                SPC_WRN(\"BIGCHERRY_PATCH_HIT patch=1268_prbe52_adaptive_mtp_wiring path=mtp_adaptive_depth contract=PRBE52-ADAPTIVE-MTP-WIRING event=depth_change old=%d new=%d drafted=%d accepted=%d seq=%d\\n\",
                        old_depth, new_depth, n_draft_actual, (int) n_accepted, (int) seq_id);
            }
        }
    }

    int32_t current_n_max(llama_seq_id seq_id) const override {
        if (params.n_min_adaptive <= 0 || seq_id < 0 || seq_id >= (llama_seq_id) adaptive_state.size() ||
                adaptive_state[seq_id].n_cur <= 0) {
            return this->n_max;
        }
        return std::min(this->n_max, adaptive_state[seq_id].n_cur);
    }

    ~common_speculative_impl_draft_mtp() override {
        if (params.n_min_adaptive <= 0 || getenv(\"BIGCHERRY_PATCH_TRACE\") == nullptr) {
            return;
        }
        for (int depth = 1; depth <= this->n_max; ++depth) {
            uint64_t rounds = 0;
            for (const auto & hist : adaptive_depth_hist) {
                if (depth < (int) hist.size()) {
                    rounds += hist[depth];
                }
            }
            if (rounds > 0) {
                SPC_WRN(\"BIGCHERRY_PATCH_HIT patch=1268_prbe52_adaptive_mtp_wiring path=mtp_adaptive_depth contract=PRBE52-ADAPTIVE-MTP-WIRING event=depth_hist depth=%d rounds=%llu\\n\",
                        depth, (unsigned long long) rounds);
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


_BASE_NMAX_ANCHOR = r"    virtual void accept\(llama_seq_id seq_id, uint16_t n_accepted, bool is_other\) = 0;\n"
_BASE_NMAX_NEW = """    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // PRBE52: static by default; adaptive MTP overrides with its current depth.
    virtual int32_t current_n_max(llama_seq_id /*seq_id*/) const { return n_max; }
"""

_NMAX_IMPL_ANCHOR = (
    r"int32_t common_speculative_n_max\(const common_speculative \* spec\) \{\n"
    r"    int32_t n_max = 0;\n\n"
    r"    if \(spec == nullptr\) \{\n"
    r"        return n_max;\n"
    r"    \}\n\n"
    r"    for \(const auto & impl : spec->impls\) \{\n"
    r"        n_max = std::max\(n_max, std::max\(0, impl->n_max\)\);\n"
    r"    \}\n\n"
    r"    return n_max;\n"
    r"\}\n"
)
_NMAX_IMPL_NEW = """int32_t common_speculative_n_max(const common_speculative * spec) {
    int32_t n_max = 0;

    if (spec == nullptr) {
        return n_max;
    }

    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, std::max(0, impl->n_max));
    }

    return n_max;
}

int32_t common_speculative_n_max(const common_speculative * spec, llama_seq_id seq_id) {
    int32_t n_max = 0;

    if (spec == nullptr) {
        return n_max;
    }

    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, std::max(0, impl->current_n_max(seq_id)));
    }

    return n_max;
}
"""

_SPEC_H_NMAX_ANCHOR = r"int32_t common_speculative_n_max\(const common_speculative \* spec\);\n"
_SPEC_H_NMAX_NEW = """int32_t common_speculative_n_max(const common_speculative * spec);
// PRBE52: effective per-sequence budget (adaptive MTP returns its current depth).
int32_t common_speculative_n_max(const common_speculative * spec, llama_seq_id seq_id);
"""

_SERVER_DRAFT_CAP_ANCHOR = (
    r"        if \(n_remaining\(\) > 0\) \{\n"
    r"            n_draft_max = std::min\(n_draft_max, n_remaining\(\) - 1\);\n"
    r"        \}\n\n"
    r"        SLT_DBG\(\*this, [^\n]*n_draft_max\);\n"
)
_SERVER_DRAFT_CAP_NEW = """        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        // PRBE52: fresh drafts and 1322 look-ahead consume the same current budget.
        n_draft_max = std::min(n_draft_max, common_speculative_n_max(spec, id));

        SLT_DBG(*this, "max possible draft: %d\\n", n_draft_max);
"""

_SERVER_INIT_ANCHOR = (
    r"            \} catch \(const std::exception & e\) \{\n"
    r"                SRV_ERR\([^\n]*e.what\(\)\);\n"
    r"                if \(params_base.speculative.has_synth\(\)\) \{\n"
    r"                    return false;\n"
    r"                \}\n"
    r"            \}\n"
)
_SERVER_INIT_NEW = """            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\\n", e.what());
                // PRBE52: requested speculation must never silently degrade to no-draft serving.
                return false;
            }
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
        edits=(Edit(id="prbe52-adaptive-arg", anchor=_ARG_ANCHOR, mode="replace", text=_ARG_NEW,
                    guard=r"LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE", rationale="Identify the n_min option by executable code tokens; strings are noise-stripped.", expect_matches=1, max_span_lines=8),),
    ),
    FilePatch(
        path="common/speculative.cpp",
        description="PRBE52 per-sequence adaptive MTP runtime wiring",
        edits=(
            Edit(id="prbe52-base-current-nmax", anchor=_BASE_NMAX_ANCHOR, mode="replace", text=_BASE_NMAX_NEW,
                 guard=r"virtual int32_t current_n_max", rationale="Expose a deterministic per-sequence effective budget.", expect_matches=1, max_span_lines=2),
            Edit(id="prbe52-nmax-overload", anchor=_NMAX_IMPL_ANCHOR, mode="replace", text=_NMAX_IMPL_NEW,
                 guard=r"common_speculative_n_max\(const common_speculative \* spec, llama_seq_id seq_id\)", rationale="Resolve the effective budget across active drafters.", expect_matches=1, max_span_lines=14),
            Edit(id="prbe52-atomic-include", anchor=_INCLUDES_ANCHOR, mode="replace", text=_INCLUDES_NEW,
                 guard=r"#include <atomic>", rationale="Thread-safe once-per-process activation evidence.", expect_matches=1, max_span_lines=3),
            Edit(id="prbe52-state", anchor=_MEMBERS_ANCHOR, mode="replace", text=_MEMBERS_NEW,
                 guard=r"adaptive_depth_hist", rationale="Identify member declarations by code only; trailing comments are noise-stripped.", expect_matches=1, max_span_lines=4),
            Edit(id="prbe52-ctor", anchor=_CTOR_ANCHOR, mode="replace", text=_CTOR_NEW,
                 guard=r"adaptive_depth_hist\.assign", rationale="Validate the post-chain-head cap and size per-sequence state.", expect_matches=1, max_span_lines=7),
            Edit(id="prbe52-begin-reset", anchor=_BEGIN_ANCHOR, mode="insert_after", text=_BEGIN_NEW,
                 guard=r"adaptive_state\.at\(seq_id\)\.reset", rationale="Reset adaptation at the request/sequence begin boundary.", expect_matches=1, max_span_lines=4),
            Edit(id="prbe52-draft-reset", anchor=_DRAFT_START_ANCHOR, mode="insert_after", text=_DRAFT_START_NEW,
                 guard=r"dp\.n_tail <= 0", rationale="Identify the MTP drafting start by its pending-h embedding batch setup, not by a pair shared with other draft implementations.", expect_matches=1, max_span_lines=3),
            Edit(id="prbe52-depth-limit", anchor=_LIMIT_ANCHOR, mode="replace", text=_LIMIT_NEW,
                 guard=r"bc_adaptive_cap <= \(int\) result\.size\(\)", rationale="Preserve the native n_max stop block for 1321, then apply the adaptive cap before chain-head continuation.", expect_matches=1, max_span_lines=6),
            Edit(id="prbe52-accept-update", anchor=_ACCEPT_ANCHOR, mode="replace", text=_ACCEPT_NEW,
                 guard=r"adaptive_state\[seq_id\]\.update", rationale="Feed the real accepted count back; anchor only on executable statements.", expect_matches=1, max_span_lines=6),
        ),
    ),
    FilePatch(
        path="common/speculative.h",
        description="PRBE52 expose effective per-sequence speculative budget",
        edits=(Edit(id="prbe52-nmax-decl", anchor=_SPEC_H_NMAX_ANCHOR, mode="replace", text=_SPEC_H_NMAX_NEW,
                    guard=r"common_speculative_n_max\(const common_speculative \* spec, llama_seq_id seq_id\)", rationale="Server look-ahead needs the controller's current per-round budget.", expect_matches=1, max_span_lines=1),),
    ),
    FilePatch(
        path="tools/server/server-context.cpp",
        description="PRBE52 cap server/ahead work and make speculative init failure fatal",
        language="none",
        edits=(
            Edit(id="prbe52-server-draft-cap", anchor=_SERVER_DRAFT_CAP_ANCHOR, mode="replace", text=_SERVER_DRAFT_CAP_NEW,
                 guard=r"fresh drafts and 1322 look-ahead", rationale="Use one budget for fresh fronts and ahead forced-front/tail work.", expect_matches=1, max_span_lines=7),
            Edit(id="prbe52-spec-init-fatal", anchor=_SERVER_INIT_ANCHOR, mode="replace", text=_SERVER_INIT_NEW,
                 guard=r"must never silently degrade", rationale="Do not serve without draft after requested speculation initialization fails.", expect_matches=1, max_span_lines=7),
        ),
    ),
]
