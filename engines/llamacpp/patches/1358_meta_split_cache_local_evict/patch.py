"""1358: a stale Meta split-state cache entry evicts itself, not the whole cache.

ggml-backend-meta.cpp memoises every tensor's split state per buffer under (tensor address, assume_sync), with a copy
of the tensor struct to detect an address that has been recycled for another tensor. On such a mismatch upstream
clears the whole cache. Graph tensor addresses are recycled between graph shapes all the time (prefill chunk, decode
step, draft verify), so one stale entry throws away the memoised state of every other tensor of the current graph and
the recursive walk recomputes it.

This patch erases only the stale entry. Every other entry is still checked against its own tensor copy when it is
looked up, exactly as before.

The change first shipped inside 1328 (auxiliary expert backend), where it was needed to keep stacked merge evaluation
from going superlinear. A two-build A/B (run aux3off, production against production + 1328 with expert offload off)
then showed the build that merely contained it prefilling 2-3% faster with identical text, which is why it is now its
own patch. On by default; BIGCHERRY_META_SPLIT_CACHE_EVICT=0 restores the whole-cache clear.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

_A_CACHE = r"""    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }
"""
_N_CACHE = r"""    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        // bigcherry 1358: graph tensor addresses are recycled between graph shapes. Clearing the whole cache here
        // throws away the memoised split state of every other tensor of the current graph; evict only the stale
        // entry. BIGCHERRY_META_SPLIT_CACHE_EVICT=0 restores the whole-cache clear.
        static const bool bc_local_evict = [] {
            const char * s = getenv("BIGCHERRY_META_SPLIT_CACHE_EVICT");
            return s == nullptr || atoi(s) != 0;
        }();
        if (bc_local_evict) {
            static bool bc_logged = false;
            if (!bc_logged && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
                bc_logged = true;
                GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1358_meta_split_cache_local_evict mechanism=local-evict entries=%zu\n",
                        buf_ctx->split_state_cache.size());
            }
            buf_ctx->split_state_cache.erase(it);
        } else {
            buf_ctx->split_state_cache.clear();
        }
        it = buf_ctx->split_state_cache.end();
    }
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-backend-meta.cpp",
        description="1358: evict only the stale split-state cache entry",
        language="none",
        edits=(
            Edit(
                id="meta-split-cache-local-evict",
                anchor=re.escape(_A_CACHE),
                mode="replace",
                text=_N_CACHE,
                guard=r"BIGCHERRY_PATCH_HIT patch=1358_meta_split_cache_local_evict",
                rationale="The one place the split-state cache reacts to a recycled tensor address.",
                expect_matches=1,
                max_span_lines=5,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_META_SPLIT_CACHE_EVICT", "0|1", "1 (on)",
           "A stale Meta split-state cache entry (recycled tensor address) is erased on its own instead of clearing "
           "the whole cache; 0 restores the whole-cache clear"),
)
