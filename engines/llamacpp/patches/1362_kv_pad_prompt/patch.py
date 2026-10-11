"""1362 (QFP53): opt-in coarse KV and k-pool padding for prompt graph reuse."""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "rejected"

_A_H_DECL = "class llama_kv_cache : public llama_memory_i {\n"
_N_H_DECL = r"""// BigCherry 1362: parsed once in llama-kv-cache.cpp; callers pass the cache size so an oversized
// BIGCHERRY_KV_PAD is disabled consistently for KV and k-pool graph sizing.
uint32_t llama_bigcherry_kv_pad(uint32_t kv_size);

class llama_kv_cache : public llama_memory_i {
"""

_A_PARSE = "static bool ggml_is_power_of_2(int n) {\n"
_N_PARSE = r"""// BigCherry 1362: one parser shared by KV and k-pool graph sizing.
uint32_t llama_bigcherry_kv_pad(uint32_t kv_size) {
    static const uint32_t bc_kv_pad = [] {
        const char * env = getenv("BIGCHERRY_KV_PAD");
        if (env == nullptr || env[0] == '\0') {
            return 0u;
        }

        uint32_t value = 0;
        bool valid = true;
        for (const char * p = env; *p != '\0'; ++p) {
            if (*p < '0' || *p > '9') {
                valid = false;
                break;
            }
            const uint32_t digit = (uint32_t) (*p - '0');
            if (value > (std::numeric_limits<uint32_t>::max() - digit) / 10u) {
                valid = false;
                break;
            }
            value = value*10u + digit;
        }

        if (!valid || value % 256u != 0) {
            LLAMA_LOG_WARN("BIGCHERRY_KV_PAD=%s ignored: expected 0 or a multiple of 256 cells\n", env);
            return 0u;
        }
        return value;
    }();

    if (bc_kv_pad > kv_size) {
        static bool bc_warned_size = false;
        if (!bc_warned_size) {
            bc_warned_size = true;
            LLAMA_LOG_WARN("BIGCHERRY_KV_PAD=%u ignored: exceeds KV size %u\n", bc_kv_pad, kv_size);
        }
        return 0u;
    }

    return bc_kv_pad;
}

static bool ggml_is_power_of_2(int n) {
"""

_A_PAD = "    const uint32_t n_pad_cur = std::max(n_pad, 256u);\n"
_N_PAD = r"""    // bigcherry 1362: BIGCHERRY_KV_PAD opt-in coarse graph padding; 0/unset keeps upstream behavior.
    const uint32_t bc_kv_pad_cur = llama_bigcherry_kv_pad(get_size());

    const char * bc_patch_trace = getenv("BIGCHERRY_PATCH_TRACE");
    if (bc_kv_pad_cur != 0 && bc_patch_trace != nullptr && atoi(bc_patch_trace) != 0) {
        static const bool bc_logged = [bc_kv_pad_cur] {
            LLAMA_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1362_kv_pad_prompt pad=%u\n", bc_kv_pad_cur);
            return true;
        }();
        (void) bc_logged;
    }

    const uint32_t n_pad_cur = std::max(n_pad, std::max(256u, bc_kv_pad_cur));
"""

_A_KPOOL_FN = "uint32_t kpool_pad(uint32_t n_pool) {\n"
_N_KPOOL_FN = "uint32_t kpool_pad(const llama_memory_hybrid_idx * mem, uint32_t n_pool) {\n"

_A_KPOOL_BODY = "    return std::max<uint32_t>(64u, GGML_PAD(n_pool + 1, 64u));\n"
_N_KPOOL_BODY = r"""    const uint32_t bc_kv_pad = llama_bigcherry_kv_pad(mem->get_mem_idx()->get_size());
    if (bc_kv_pad == 0) {
        return std::max<uint32_t>(64u, GGML_PAD(n_pool + 1, 64u));
    }

    const uint32_t kpool = mem->get_kpool();
    GGML_ASSERT(kpool > 0);
    const uint32_t bc_pool_quantum = std::max<uint32_t>(64u, GGML_PAD(bc_kv_pad / kpool, 64u));
    const uint64_t bc_n_pool = (uint64_t) n_pool + 1u;
    const uint64_t bc_padded = (bc_n_pool + bc_pool_quantum - 1u) / bc_pool_quantum * bc_pool_quantum;
    GGML_ASSERT(bc_padded <= UINT32_MAX);
    return std::max<uint32_t>(bc_pool_quantum, (uint32_t) bc_padded);
"""

_A_KPOOL_NEW = "    st.n_new_g = std::max({st.n_new, 1u, std::min({bound, kpool_pad(st.n_pool_real) - 1, n_pool_max})});\n"
_N_KPOOL_NEW = "    st.n_new_g = std::max({st.n_new, 1u, std::min({bound, kpool_pad(mem, st.n_pool_real) - 1, n_pool_max})});\n"

_A_KPOOL_GET = "    return kpool_pad(kpool_cur().n_pool_real);\n"
_N_KPOOL_GET = "    return kpool_pad(mem, kpool_cur().n_pool_real);\n"

_A_KPOOL_ASSERT = "    GGML_ASSERT(n_pool == kpool_pad(st.n_pool_real));\n"
_N_KPOOL_ASSERT = "    GGML_ASSERT(n_pool == kpool_pad(mem, st.n_pool_real));\n"

PATCHES = [
    FilePatch(
        path="src/llama-kv-cache.h",
        description="1362: expose one shared BIGCHERRY_KV_PAD accessor",
        language="none",
        edits=(
            Edit(
                id="kv-pad-shared-accessor-decl",
                anchor=re.escape(_A_H_DECL),
                mode="replace",
                text=_N_H_DECL,
                guard=r"uint32_t llama_bigcherry_kv_pad\(uint32_t kv_size\);",
                rationale="The KV and k-pool paths must consume the same parsed environment value.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="src/llama-kv-cache.cpp",
        description="1362: optional coarse KV graph padding with shared parser",
        language="none",
        edits=(
            Edit(
                id="kv-pad-shared-accessor-def",
                anchor=re.escape(_A_PARSE),
                mode="replace",
                text=_N_PARSE,
                guard=r"uint32_t llama_bigcherry_kv_pad\(uint32_t kv_size\) \{",
                rationale="Parse BIGCHERRY_KV_PAD once and centralize cache-size validation for every consumer.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="kv-pad-prompt",
                anchor=re.escape(_A_PAD),
                mode="replace",
                text=_N_PAD,
                guard=r"bigcherry 1362: BIGCHERRY_KV_PAD opt-in coarse graph padding",
                rationale="The unique get_n_kv padding statement; preserving the surrounding GGML_PAD/clamp logic keeps cache bounds unchanged.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="src/llama-memory-hybrid-idx.cpp",
        description="1362: tie k-pool graph padding to the KV padding quantum",
        language="none",
        edits=(
            Edit(
                id="kpool-pad-signature",
                anchor=re.escape(_A_KPOOL_FN),
                mode="replace",
                text=_N_KPOOL_FN,
                guard=r"uint32_t kpool_pad\(const llama_memory_hybrid_idx \* mem, uint32_t n_pool\) \{",
                rationale="The pool padding helper needs the model kpool and indexer cache size.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="kpool-pad-quantum",
                anchor=re.escape(_A_KPOOL_BODY),
                mode="replace",
                text=_N_KPOOL_BODY,
                guard=r"const uint32_t bc_pool_quantum = std::max<uint32_t>\(64u, GGML_PAD\(bc_kv_pad / kpool, 64u\)\);",
                rationale="Use the same opt-in cell bucket for the Qwen4Exp/GLM5 pooled-block graph dimension while retaining exact upstream behavior when disabled.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="kpool-new-bound",
                anchor=re.escape(_A_KPOOL_NEW),
                mode="replace",
                text=_N_KPOOL_NEW,
                guard=r"kpool_pad\(mem, st\.n_pool_real\) - 1",
                rationale="The new-pool stable bound must use the same padded pool capacity as the graph tensors.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="kpool-count",
                anchor=re.escape(_A_KPOOL_GET),
                mode="replace",
                text=_N_KPOOL_GET,
                guard=r"return kpool_pad\(mem, kpool_cur\(\)\.n_pool_real\);",
                rationale="get_n_kpool is the graph-shape source used by Qwen4Exp and GLM5-Next.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="kpool-input-assert",
                anchor=re.escape(_A_KPOOL_ASSERT),
                mode="replace",
                text=_N_KPOOL_ASSERT,
                guard=r"GGML_ASSERT\(n_pool == kpool_pad\(mem, st\.n_pool_real\)\);",
                rationale="set_input_kpool must fill exactly the same padded width that graph construction allocated.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc(
        "BIGCHERRY_KV_PAD",
        "cells",
        "0",
        "opt-in KV and k-pool graph padding; nonzero must be a multiple of 256 and no larger than the KV cache",
    ),
)
