"""1362 (QFP53): opt-in coarser KV padding for prompt graph reuse."""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_A_PAD = "    const uint32_t n_pad_cur = std::max(n_pad, 256u);\n"
_N_PAD = r"""    // bigcherry 1362: BIGCHERRY_KV_PAD opt-in coarse graph padding; 0/unset keeps upstream behavior.
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

    uint32_t bc_kv_pad_cur = bc_kv_pad;
    const uint32_t bc_kv_size = get_size();
    if (bc_kv_pad_cur > bc_kv_size) {
        static const bool bc_warned_size = [bc_kv_pad_cur, bc_kv_size] {
            LLAMA_LOG_WARN("BIGCHERRY_KV_PAD=%u ignored: exceeds KV size %u\n", bc_kv_pad_cur, bc_kv_size);
            return true;
        }();
        (void) bc_warned_size;
        bc_kv_pad_cur = 0;
    }

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

PATCHES = [
    FilePatch(
        path="src/llama-kv-cache.cpp",
        description="1362: optional coarse KV graph padding for prompt graph reuse",
        language="none",
        edits=(
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
]

ENV_DOCS = (
    EnvDoc(
        "BIGCHERRY_KV_PAD",
        "cells",
        "0",
        "opt-in get_n_kv graph padding; nonzero must be a multiple of 256 and no larger than the KV cache",
    ),
)
