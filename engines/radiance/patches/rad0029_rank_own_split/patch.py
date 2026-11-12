"""rad0029 (VTP33): in an uneven dense world, every rank gets its own VRAM split.

The VRAM budget is one split for every rank: the smallest card, the largest weights and the dearest cache. One block
table serves every rank, so the common token count is the least that any rank pages. Only ranks whose cache is
cheaper per token are re-split (under expert_tiered), so the rank that sets the token count keeps the common split even
when it has memory to spare. At TP3 over XTX, XTX and R9700, the R9700 holds 2 of the 4 KV heads, so it pages the
fewest tokens. Its card is 8 GB larger, and with a split that gives it fewer weights its weights pool has unused room.
Both stayed unused: after a load, about 8 GiB of the R9700 sat free.

When every rank's model is dense (no expert plane), its weights all fit the common weights pool and no card limit was
given (--gpu-memory-mib) on a discrete card (under any --placement; in a dense world it replaces the cheaper-rank
re-split, which there only shrinks a KV pool and frees nothing), each rank's weights pool becomes its own weights. Its KV pool takes the
common pool, plus what its weights pool gave back, plus the free memory it measured above the smallest card's. The
common token count is then the least any rank pages in its own pool, and every rank's KV pool is cut to the least whole
MiB that pages that count. When that count is no larger than the common split's, nothing changes.
Source-pinned to radiance v1.17.0 (19d4e8fe), after rad0028.
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

_ANCHOR = """                         ? fmt(", and %lld MiB stay unclaimed", (long long)rs.unusable_mib).c_str()
                         : "");
        }
    }

    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
"""

_TEXT = """                         ? fmt(", and %lld MiB stay unclaimed", (long long)rs.unusable_mib).c_str()
                         : "");
        }

        /* BC_VTP33_OWN_SPLIT (rad0029): an uneven dense world gives every rank its own split. It
         * replaces the cheaper-rank re-split above, which in a dense world only shrinks a cheaper
         * rank's KV pool (its weights pool cannot grow past its weights) and frees nothing. Each weights pool holds its own rank's weights; each KV
         * pool takes the common pool, what its weights pool gave back and the free memory it
         * measured above the smallest card's. The common token count is the least any rank pages
         * in its own pool, and each KV pool is cut to the least whole MiB that pages it. */
        {
            bool own = ranks_.size() > 1 && facts.size() == ranks_.size() &&
                       cfg_.gpu_memory_mib <= 0;
            int64_t free_min = -1, statics_max = 0;
            for (const VramFacts& f : facts) {
                own = own && f.experts == 0 && !f.unified;
                if (free_min < 0 || f.free < free_min) free_min = f.free;
                statics_max = std::max(statics_max, f.statics);
            }
            own = own && (statics_max + kMiB - 1) / kMiB <= cfg_.vram_weights_mib;
            std::vector<std::unique_ptr<KVManager>> own_kv(ranks_.size());
            std::vector<int64_t> own_w(ranks_.size(), cfg_.vram_weights_mib);
            std::vector<int64_t> own_pool(ranks_.size(), cfg_.vram_kv_mib);
            int64_t T_common = -1, T_own = -1;
            for (size_t i = 0; own && i < ranks_.size(); ++i) {
                own_w[i] = std::max<int64_t>((facts[i].statics + kMiB - 1) / kMiB, 1);
                own_pool[i] = cfg_.vram_kv_mib + (cfg_.vram_weights_mib - own_w[i]) +
                              std::max<int64_t>(facts[i].free - free_min, 0) / kMiB;
                own_kv[i] = std::make_unique<KVManager>();
                own_kv[i]->set_drafter_window_snapshot(drafter_window_snap(ranks_[i]->program));
                if (declare_ring_windows(*own_kv[i], ranks_[i]->program) != RAD_OK ||
                    own_kv[i]->plan_groups(ranks_[i]->program.kv_groups, cfg_) != RAD_OK) {
                    own = false;
                    break;
                }
                /* The common split's count: what every rank pages in the common pool, which is
                 * what the cheaper-rank re-split (if any) also settled on. */
                const int64_t tc = own_kv[i]->paged_tokens(cfg_.vram_kv_mib * kMiB);
                const int64_t to = own_kv[i]->paged_tokens(own_pool[i] * kMiB);
                if (tc == KVManager::kUnboundedTokens || to == KVManager::kUnboundedTokens) continue;
                if (T_common < 0 || tc < T_common) T_common = tc;
                if (T_own < 0 || to < T_own) T_own = to;
            }
            if (own && T_own > T_common && T_common > 0) {
                std::vector<int64_t> need(ranks_.size(), cfg_.vram_kv_mib);
                for (size_t i = 0; own && i < ranks_.size(); ++i) {
                    need[i] = own_kv[i]->pool_mib_for_tokens(T_own, own_pool[i]);
                    if (need[i] < 0) own = false;
                }
                for (size_t i = 0; own && i < ranks_.size(); ++i) {
                    rank_cfg_[i].vram_weights_mib = own_w[i];
                    rank_cfg_[i].vram_kv_mib      = need[i];
                    RAD_INFO("kv: rank %d's own split: weights pool %lld MiB (its weights), KV pool "
                             "%lld MiB of the %lld it could take, so every rank pages %lld tokens "
                             "instead of the common split's %lld", ranks_[i]->index,
                             (long long)own_w[i], (long long)need[i], (long long)own_pool[i],
                             (long long)T_own, (long long)T_common);
                }
                if (own) kv_planned_tokens_ = T_own;
            }
        }
    }

    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
"""

BRINGUP = FilePatch(path="core/engine_bringup.cpp", language="none", description="VTP33 every rank its own split",
                    edits=(
    Edit(
        id="own-split",
        anchor=re.escape(_ANCHOR),
        mode="replace",
        text=_TEXT,
        guard=re.escape("BC_VTP33_OWN_SPLIT"),
        expect_matches=1,
        max_span_lines=_ANCHOR.count("\n") + 1,
        rationale="long-anchor: the insertion closes the cheaper-rank loop and its block, so the anchor spans both closing braces and the next loop header. VTP33: the rank that sets the common token count uses its own card's memory",
    ),
))

PATCHES = (BRINGUP,)
