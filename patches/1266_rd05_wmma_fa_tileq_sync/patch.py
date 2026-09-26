"""PRBE110/RD05: extract only the WMMA flash-attention tile_Q reuse barriers.

This is the RD05 correctness sub-piece of rejected package 1203. RD06's
RDNA4 configuration expansion and RD07's Q6_K MMQ work are intentionally
excluded. The activation marker is local BigCherry instrumentation.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

_K00_SYNC_OLD = """        if (np > 1) {
            __syncthreads();
        }"""

_K00_SYNC_NEW = """        // The tile_Q buffer is reused for the next k00 iteration, so all warps must sync here
        // before its data is overwritten. With np > 1 only some warps read back, but they all write.
        if (np > 1 || k00 + nbatch_combine < DV/2) {
            __syncthreads();
        }"""

_KBC_ANCHOR_OLD = """        }

        kbc += iter_k;
        kbc -= kbc % iter_k;"""

_KBC_ANCHOR_NEW = """        }

        // The next process_tile call reuses the tile_Q buffer for its Q/K tiles, so all warps must
        // have finished reading the combined results before any of them starts the next call.
        // (With np == 1 the end-of-k00 barrier does not fire, so this is required for correctness.)
        __syncthreads();

        kbc += iter_k;
        kbc -= kbc % iter_k;"""

_FATTN_INCLUDES_OLD = """#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
"""

_FATTN_INCLUDES_NEW = """#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"

#include <atomic>
"""

_DISPATCH_OLD = """        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;"""

_DISPATCH_NEW = """        case BEST_FATTN_KERNEL_MMA_F16: {
            // bigcherry: PRBE110/RD05 activation evidence; not part of the source port.
            if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
                static std::atomic_flag bigcherry_rd05_logged = ATOMIC_FLAG_INIT;
                if (!bigcherry_rd05_logged.test_and_set(std::memory_order_relaxed)) {
                    GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1266_rd05_wmma_fa_tileq_sync path=wmma_f16_dispatch contract=PRBE110-RD05-WMMA-TILEQ-SYNC\\n");
                }
            }
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
        }"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/fattn-mma-f16.cuh",
        description="RD05 tile_Q reuse synchronization only",
        edits=(
            Edit(
                id="rd05-k00-sync",
                anchor=_re.escape(_K00_SYNC_OLD),
                mode="replace",
                text=_K00_SYNC_NEW,
                guard=_re.escape("if (np > 1 || k00 + nbatch_combine < DV/2) {"),
                rationale="Prevent the next k00 iteration from overwriting tile_Q before all warps finish reading it.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="rd05-kbc-sync",
                anchor=_re.escape(_KBC_ANCHOR_OLD),
                mode="replace",
                text=_KBC_ANCHOR_NEW,
                guard=_re.escape("The next process_tile call reuses the tile_Q buffer for its Q/K tiles"),
                rationale="Synchronize all warps before the next process_tile call reuses tile_Q.",
                expect_matches=1,
                max_span_lines=5,
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/fattn.cu",
        description="RD05-only activation marker at the real WMMA-F16 dispatch",
        edits=(
            Edit(
                id="rd05-atomic-include",
                anchor=_re.escape(_FATTN_INCLUDES_OLD),
                mode="replace",
                text=_FATTN_INCLUDES_NEW,
                guard=r"#include <atomic>",
                rationale="The once-per-process RD05 trace marker uses std::atomic_flag.",
                expect_matches=1,
                max_span_lines=7,
            ),
            Edit(
                id="rd05-activation-marker",
                anchor=_re.escape(_DISPATCH_OLD),
                mode="replace",
                text=_DISPATCH_NEW,
                guard=_re.escape("BIGCHERRY_PATCH_HIT patch=1266_rd05_wmma_fa_tileq_sync path=wmma_f16_dispatch contract=PRBE110-RD05-WMMA-TILEQ-SYNC"),
                rationale="Prove the existing WMMA-F16 launch path containing the synchronization fix executed without carrying RD06 gating.",
                expect_matches=1,
                max_span_lines=4,
            ),
        ),
    ),
]
