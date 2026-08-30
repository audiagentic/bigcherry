"""RD39+RD40+RD41+RD42 (AMD-STREAM-001..004): honor the active HIP stream in
the non-split matmul path, give every (device,stream) pair its own cuBLAS
handle, isolate concurrent graph-branch scratch in a dedicated buffer, and
overlap the MoE shared expert on an auxiliary stream during decode.

Provenance (group 'rdna-boosts' patches are external backports; the
machine-readable PROVENANCE dict below is cross-checked against
external-sources.toml by tools/tests/test_external_sources.py):

  source:            amd-ecosystem-llama-cpp
  repo:              https://github.com/AMD-Ecosystem/llama.cpp
  locator:           PR #36 (fork tracks per-PR, not per-branch -- see the
                     source's own registry notes)
  merge commit:      367c4d04f409f89db2512d4d035915ccaa84c42d
  title:             "ggml-cuda: overlap the MoE shared expert on a
                     separate stream"
  reviewed:          2026-08-20 (RD28-RD53 validation pass); merged, not
                     ancestral to ggml-org/llama.cpp mainline (fork-only
                     work) and not ancestral to our b10502 pin.

Why one patch for four plan items:
  RD39 (honor stream), RD40 (per-stream BLAS handles) and RD41 (dedicated
  branch scratch) are declared prerequisites of each other and of RD42 (the
  MoE overlap payload) in the plan items themselves (RD40/RD41 depend on
  RD39; RD42 depends on all three) -- they were never meant to be enabled
  independently. They are also literally one PR (#36, 2 files, +203/-39)
  whose hunks are causally coupled: RD41's scratch-buffer placement code is
  what RD42's shared-expert branch grouping feeds into. Splitting them into
  four independently-anchored patches would require each intermediate
  patch to leave the graph optimizer in a half-migrated, uncompilable
  state. Ported as one net patch, consistent with this project's existing
  practice for causally-coupled multi-commit fork diffs (see 1202's
  seven-commit net range and 1203's mixed RD05/06/07 commit).

What it does (performance, gated to decode / small-batch MoE):
  ggml_cuda_op_mul_mat previously hardcoded stream 0 on the non-split path,
  so a matmul assigned to an auxiliary stream still ran on the main stream;
  the single per-device cuBLAS handle was also shared across streams, and
  its workspace corrupts concurrent GEMMs sharing it. Both are fixed here
  (RD39/RD40). The existing (pre-patch, from upstream PR #16991) attention
  QKV concurrency interleaved branch nodes in-place to extend their
  lifetimes against ggml-alloc reuse; that interleaving is fragile (order
  can desync from allocator expectations). RD41 replaces it with a
  dedicated scratch buffer sized to the largest concurrent region and
  reused across layers, so branch nodes never alias each other or tensors
  read across the region. RD42 uses this same infrastructure to detect the
  `join = add(ffn_moe_out, ffn_shexp)` diamond in MoE-with-shared-expert
  models and fork the shared expert onto one aux stream, joined at the
  add -- gated to small batches (<= MMVQ_MAX_BATCH_SIZE tokens), the
  memory-bound regime where the routed branch leaves the GPU underutilized
  for the shared expert to run alongside. Upstream-fork measured
  (Qwen3.6-35B-A3B Q4_K_M, tg128): +7.4%. Output is byte-identical to
  sequential; requires CUDA/HIP graphs and a single GPU (both already true
  of the pre-existing QKV concurrency this shares infrastructure with).

Porting notes:
  - Both anchor regions (ggml/src/ggml-cuda/common.cuh struct fields/
    methods; ggml/src/ggml-cuda/ggml-cuda.cu destructor, graph-evaluate
    loop, graph_optimize) are BYTE-IDENTICAL between the fork's PR base
    and our pinned b10502 tree -- this region is untouched by any other
    BigCherry framework or RD patch (verified 2026-08-20 by direct
    comparison against vendor/llama.cpp).
  - No new symbols beyond the standard library (<unordered_set>, added to
    ggml-cuda.cu's includes).
  - GGML_CUDA_GRAPH_OPT stays opt-in (env var) on this patch alone; the
    RDNA3.5 default-on flip is RD44 (separate patch, PR #56) and the
    join-node fusion-safety fix (needed once graph-opt is more broadly
    exercised) is RD43 (separate patch, PR #71) -- apply this patch
    before either.

Hardware status (this patch's own acceptance, from the plan items):
  RD39-RD41 need correctness verification (output equality under forced
  concurrency) before RD42's performance claim is meaningful; RD42 itself
  needs profiler proof of actual overlap plus >1% E2E gain under a
  workload predicate guaranteeing independent branches. All UNVALIDATED on
  BigCherry hardware as of porting -- STATE stays "untested".

Isolation and promotion (first-sweep policy, matching existing RD items):
  - GROUP 'rdna-boosts' + STATE 'untested' keeps this OUT of the
    production 'framework' and 'validated-enhancements' patch-sets.
  - Promotion requires the correctness + overlap-profiler evidence above,
    not just a build-clean patch.
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

PROVENANCE = {
    "source-id": "amd-ecosystem-llama-cpp",
    "plan-item": "RD39/RD40/RD41/RD42",
    "fork-commit": "367c4d04f409f89db2512d4d035915ccaa84c42d",
    "fork-commit-title": "ggml-cuda: overlap the MoE shared expert on a separate stream",
    "snapshot-head": "58ab0a5f2ce3f426d657d55647846b03fbc1a20b",
    "snapshot-base": "58ab0a5f2ce3f426d657d55647846b03fbc1a20b",
    "adaptations": [
        "None -- both target files are byte-identical between the fork's "
        "PR #36 base and our b10502 pin in the anchored regions; ported "
        "verbatim.",
    ],
}


# =========================================================================
# ggml/src/ggml-cuda/common.cuh
# =========================================================================

# --- hunk 1: per-(device,stream) cuBLAS handle array + new scratch fields
_STRUCT_OLD = """    cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { { nullptr } };
    cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES] = {nullptr};

    int curr_stream_no = 0;
"""

_STRUCT_NEW = """    cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { { nullptr } };
    // one cuBLAS handle per (device, stream): the handle carries a workspace that must not be shared
    // by concurrent streams, otherwise overlapped GEMMs corrupt each other's results
    cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { { nullptr } };

    int curr_stream_no = 0;
"""

# --- hunk 2: dedicated concurrent-branch scratch buffer fields
_SCRATCH_FIELD_OLD = """    ggml_cuda_stream_context concurrent_stream_context;

    ~ggml_backend_cuda_context();
"""

_SCRATCH_FIELD_NEW = """    ggml_cuda_stream_context concurrent_stream_context;

    // dedicated buffer for the branches of overlapped concurrent regions (attention QKV, MoE shared
    // expert), reused across layers so their scratch never aliases tensors read across the region
    ggml_backend_buffer_t concurrent_scratch      = nullptr;
    size_t                concurrent_scratch_size = 0;

    ~ggml_backend_cuda_context();
"""

# --- hunk 3: RECONCILED 2026-08-31 (b10687->b10692 bump) -- REMOVED, not
# just re-anchored. Upstream PR #26574 ("ggml-cuda: provide static
# workspace for cuBLAS handles", merged 2026-08-20, one day after our
# b10502 base) independently landed per-(device,stream) cuBLAS handles as
# part of "account for concurrent streams when using GGML_CUDA_GRAPH_OPT" --
# the exact problem RD39/RD40 targeted -- and upstream's version is MORE
# complete than ours (it also isolates the cuBLAS workspace per stream,
# via cublasSetStream/cublasSetWorkspace, which this patch never
# attempted). Verified live: cublas_handle() in the current pinned
# ggml-cuda/common.cuh already indexes cublas_handles[device][curr_stream_no]
# with its own workspace management; re-applying our narrower version
# would be a regression, not a fix. The struct-field array-widening hunk
# (immediately above) reports "already-applied" against upstream's own
# 2D array for the same reason and is deliberately left in place
# (harmless no-op, documents original intent); this hunk is removed
# outright since keeping a stale anchor here would just re-break on the
# next drift. See RD39/RD40 plan items and HI154 for the full trail.


# =========================================================================
# ggml/src/ggml-cuda/ggml-cuda.cu
# =========================================================================

# --- hunk 4: new include for the MoE-overlap reachability set
_INCLUDE_OLD = """#include <cstdlib>
#include <string>
#include <vector>
"""

_INCLUDE_NEW = """#include <cstdlib>
#include <string>
#include <unordered_set>
#include <vector>
"""

# --- hunk 5: free the concurrent-branch scratch buffer in the destructor.
# Upstream PR #26574 (static per-stream cuBLAS workspace, merged
# 2026-08-20, landed inside the b10687->b10692 bump window) independently
# rewrote this destructor to already destroy per-(device,stream) handles
# AND free cublas_workspaces[i][j] itself -- the cublas_handle-widening
# half of this patch (RD39/RD40) is superseded by that PR (see the
# _HANDLE_METHOD_OLD/NEW removal above). What upstream's destructor does
# NOT know about is concurrent_scratch, which is ours alone (RD41/RD42's
# dedicated scratch buffer for the MoE shared-expert overlap) -- so this
# hunk now only layers that one free onto upstream's real current body,
# anchored on its tail to survive the handle/workspace rewrite.
_DTOR_OLD = """            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}
"""

_DTOR_NEW = """            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
    if (concurrent_scratch != nullptr) {
        ggml_backend_buffer_free(concurrent_scratch);
    }
}
"""

# --- hunk 6 (RD39, occurrence 1): join-node handler's else-branch fallback.
# Anchor stops right before the GGML_LOG_DEBUG call: its format string is a
# string literal, which strip_noise blanks to spaces before anchor matching
# (see csource.strip_noise) -- an anchor spanning it would never match. The
# log line and closing braces are left untouched by this edit.
_STREAM_FALLBACK_1_OLD = """                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
"""

_STREAM_FALLBACK_1_NEW = """                    } else {
                        // region nodes not mapped to a concurrent stream run on the main stream:
                        // this keeps the routed branch on the main stream while only the shared
                        // expert forks off (the vLLM shared-expert model)
                        auto it = concurrent_event->stream_mapping.find(node);
                        cuda_ctx->curr_stream_no = it != concurrent_event->stream_mapping.end() ? it->second : 0;
"""

# --- hunk 7 (RD39, occurrence 2): post-fusion resume fallback. Same
# string-literal reason for stopping the anchor before GGML_LOG_DEBUG.
_STREAM_FALLBACK_2_OLD = """                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
"""

_STREAM_FALLBACK_2_NEW = """                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        auto it = concurrent_event->stream_mapping.find(node);
                        cuda_ctx->curr_stream_no = it != concurrent_event->stream_mapping.end() ? it->second : 0;
"""

# --- hunk 8: declare the per-event branch-node lists alongside
# concurrent_node_ranges. Anchor starts after the "// store {fork_idx,
# join_idx}" comment (strip_noise blanks comments -- see hunk 6's note).
_RANGES_DECL_OLD = """    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
"""

_RANGES_DECL_NEW = """    std::vector<std::pair<int, int>> concurrent_node_ranges;

    // per-event lists of concurrent branch nodes to place in the dedicated scratch buffer (below),
    // so the branches are mutually disjoint and disjoint from tensors read across the region -
    // this replaces the fragile node interleaving that ggml-alloc/execution order can desync
    std::vector<std::vector<const ggml_tensor *>> concurrent_groups;

    for (const auto & [root_node, count] : fan_out) {
"""

# --- hunk 9a: drop the three declarations the in-place interleave used
# (comment-free span -- no wildcarding needed).
_DECLS_OLD = """                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
"""

_DECLS_NEW = """                int total_branch_nodes = 0;
"""

# --- hunk 9b: replace the in-place interleave loop with scratch-buffer
# grouping, then (after the fan_out loop closes) detect the MoE
# shared-expert diamond and allocate/place the dedicated scratch buffer
# for every concurrent group.
#
# The anchor deliberately starts at `concurrent_node_ranges.emplace_back(...)`
# (comment-free code, unaffected by hunk 9a) rather than the top of the
# `if (join_node)` block: everything between there and here (the
# total_branch_nodes bookkeeping, its GGML_LOG_DEBUG string literal, the
# "Save the original order" comment, concurrent_event.original_order/
# concurrent_events code) is untouched by this patch, and string literals
# blank to spaces under strip_noise (see hunk 6's note) just like comments
# do -- spanning it would only add anchor-matching risk for no reason.
#
# Two pre-existing comments DO fall inside the span this edit touches (the
# 4-line "interleave tensors..." block and the 1-line "append all empty
# nodes" line); both are replaced with `[ ]{min,max}` wildcards matching
# their strip_noise-blanked spaces, same technique as patch 1206.
_EMPLACE_LINE = """                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

"""

_WHILE_HEAD = """                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

"""

_WHILE_TAIL = """                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}
"""


def _comment_wildcard(indent: int, lines: int, max_len: int = 160) -> str:
    """Regex matching `lines` strip_noise-blanked ``//`` comment lines.

    strip_noise blanks a line comment to spaces of the same length,
    preserving the newline (see csource.strip_noise); the leading
    whitespace before ``//`` is untouched. `indent` is that leading
    whitespace's exact width; the rest of the (now-blank) line is a
    variable-length run of spaces, matched by a bounded range rather than
    the exact original length so the anchor survives re-wrapping of the
    comment text on a future re-port.
    """
    line = " " * indent + r"[ ]{1,%d}" % max_len + r"\n"
    return line * lines


_MOE_OVERLAP_ANCHOR = (
    re.escape(_EMPLACE_LINE)
    + _comment_wildcard(16, 4)
    + re.escape(_WHILE_HEAD)
    + _comment_wildcard(20, 1)
    + re.escape(_WHILE_TAIL)
)

_MOE_OVERLAP_NEW = """                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // place all branch nodes in the dedicated scratch buffer (below) instead of
                // interleaving them: ggml-alloc then keeps the branches mutually disjoint and
                // disjoint from the fork output that every branch reads concurrently
                std::vector<const ggml_tensor *> group;
                for (const auto & branch_nodes : nodes_per_branch) {
                    for (const ggml_tensor * n : branch_nodes) {
                        group.push_back(n);
                    }
                }
                concurrent_groups.push_back(std::move(group));
            }
        }
    }

    // MoE shared-expert overlap: run the shared expert on a separate stream, overlapped with the
    // routed experts. fork = the FFN-input norm feeding both branches, join = ggml_add(ffn_moe_out,
    // ffn_shexp*). Operands are matched by the names set via cb() in the model graph. Decode only
    // (gated below): prefill is compute-bound and gains nothing from the overlap.
    const auto reach_backward = [](const ggml_tensor * start) {
        std::unordered_set<const ggml_tensor *> seen;
        std::vector<const ggml_tensor *> stack = { start };
        while (!stack.empty()) {
            const ggml_tensor * t = stack.back();
            stack.pop_back();
            if (!t || seen.count(t)) {
                continue;
            }
            seen.insert(t);
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (t->src[s]) {
                    stack.push_back(t->src[s]);
                }
            }
        }
        return seen;
    };

    for (int join_idx = 0; join_idx < cgraph->n_nodes; ++join_idx) {
        ggml_tensor * join_node = cgraph->nodes[join_idx];
        if (join_node->op != GGML_OP_ADD) {
            continue;
        }

        // Only overlap during decode (single token). Overlapping the shared expert only helps when
        // the routed branch leaves the GPU underutilized for it to run alongside; that is the case in
        // decode (batch 1, latency/occupancy-bound) but not in prefill, where the routed matmuls are
        // large and already saturate the GPU, so the overlap adds contention without a speedup.
        if (ggml_nrows(join_node) > 1) {
            continue;
        }

        ggml_tensor * routed_out = nullptr;
        ggml_tensor * shexp_out  = nullptr;
        for (int s = 0; s < 2; ++s) {
            ggml_tensor * x = join_node->src[s];
            ggml_tensor * y = join_node->src[1 - s];
            if (x && y && strstr(x->name, "ffn_moe_out") && strstr(y->name, "ffn_shexp")) {
                routed_out = x;
                shexp_out  = y;
            }
        }
        if (!routed_out || !shexp_out) {
            continue;
        }

        const std::unordered_set<const ggml_tensor *> reach_routed = reach_backward(routed_out);
        const std::unordered_set<const ggml_tensor *> reach_shexp  = reach_backward(shexp_out);

        // fork = highest-index node reachable from both branches (the ffn_norm output)
        int fork_idx = -1;
        for (const ggml_tensor * t : reach_routed) {
            if (!reach_shexp.count(t)) {
                continue;
            }
            auto it = node_indices.find(t);
            if (it != node_indices.end() && it->second < join_idx && it->second > fork_idx) {
                fork_idx = it->second;
            }
        }
        if (fork_idx < 0) {
            continue;
        }

        bool overlaps = false;
        for (const auto & [start, end] : concurrent_node_ranges) {
            if (!(join_idx < start || fork_idx > end)) {
                overlaps = true;
            }
        }
        if (overlaps) {
            continue;
        }

        // partition the region (fork_idx, join_idx): shared-expert nodes -> stream 2, routed -> 1
        std::vector<std::vector<const ggml_tensor *>> nodes_per_branch(2);
        for (int i = fork_idx + 1; i < join_idx; ++i) {
            const ggml_tensor * n = cgraph->nodes[i];
            const int branch = reach_shexp.count(n) ? 1 : 0;
            nodes_per_branch[branch].push_back(n);
        }
        if (nodes_per_branch[0].empty() || nodes_per_branch[1].empty()) {
            continue;
        }

        // vLLM shared-expert model: the routed experts stay on the main stream and only the shared
        // expert forks onto a single aux stream, joined at the add. Keeping the large routed branch
        // on the main stream avoids migrating it and needs only one fork/join.
        ggml_cuda_concurrent_event concurrent_event(1);
        concurrent_event.join_node = join_node;
        for (const ggml_tensor * n : nodes_per_branch[1]) {
            concurrent_event.stream_mapping[n] = 1;
        }

        const ggml_tensor * fork_node = cgraph->nodes[fork_idx];
        concurrent_event.original_order.reserve(join_idx - fork_idx - 1);
        for (int i = fork_idx + 1; i < join_idx; ++i) {
            concurrent_event.original_order.push_back(cgraph->nodes[i]);
        }

        std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
        if (concurrent_events.find(fork_node) != concurrent_events.end()) {
            continue;
        }
        concurrent_events.emplace(fork_node, std::move(concurrent_event));
        GGML_LOG_DEBUG("Adding shared-expert stream at node %s %p\\n", fork_node->name, fork_node);
        concurrent_node_ranges.emplace_back(fork_idx, join_idx);

        // the shared-expert nodes get a dedicated buffer (below), so the graph order is left intact
        // and no interleaving is needed to keep the branch non-overlapping
        concurrent_groups.push_back(nodes_per_branch[1]);
    }

    // Place every concurrent branch (attention QKV and MoE shared-expert) in a dedicated buffer so
    // its nodes never share an address with each other or with tensors read across the region (which
    // ggml-alloc could otherwise recycle, corrupting concurrent reads). Layers run sequentially, so
    // one buffer sized to the largest region is reused across all of them; within a region each node
    // gets a distinct offset so the concurrent scratch stays disjoint.
    if (!concurrent_groups.empty()) {
        const size_t alignment = 128;

        const auto group_footprint = [&](const std::vector<const ggml_tensor *> & group) {
            size_t off = 0;
            for (const ggml_tensor * n : group) {
                if (is_noop(n) || n->view_src != nullptr) {
                    continue;
                }
                off += GGML_PAD(ggml_nbytes(n), alignment);
            }
            return off;
        };

        size_t needed = 0;
        for (const auto & group : concurrent_groups) {
            needed = std::max(needed, group_footprint(group));
        }

        if (needed > 0) {
            if (cuda_ctx->concurrent_scratch == nullptr || cuda_ctx->concurrent_scratch_size < needed) {
                if (cuda_ctx->concurrent_scratch != nullptr) {
                    ggml_backend_buffer_free(cuda_ctx->concurrent_scratch);
                }
                cuda_ctx->concurrent_scratch      = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(cuda_ctx->device), needed);
                cuda_ctx->concurrent_scratch_size = needed;
            }

            char * const base = (char *) ggml_backend_buffer_get_base(cuda_ctx->concurrent_scratch);
            for (const auto & group : concurrent_groups) {
                size_t off = 0;
                for (const ggml_tensor * cn : group) {
                    if (is_noop(cn) || cn->view_src != nullptr) {
                        continue;
                    }
                    ggml_tensor * n = const_cast<ggml_tensor *>(cn);
                    n->data   = base + off;
                    n->buffer = cuda_ctx->concurrent_scratch;
                    off += GGML_PAD(ggml_nbytes(n), alignment);
                }
            }
        }
    }
}
"""


PATCH_COMMON_CUH = FilePatch(
    path="ggml/src/ggml-cuda/common.cuh",
    description="Per-(device,stream) cuBLAS handles and a dedicated "
                "concurrent-branch scratch buffer (amd-ecosystem PR #36 / "
                "RD39-RD42)",
    edits=(
        Edit(
            id="rd3942-cublas-handle-array",
            anchor=re.escape(_STRUCT_OLD),
            rationale="widen cublas_handles to [device][stream] so "
                      "concurrent streams never share a handle workspace",
            mode="replace",
            text=_STRUCT_NEW,
            guard=r"cublasHandle_t cublas_handles\[GGML_CUDA_MAX_DEVICES\]\[GGML_CUDA_MAX_STREAMS\]",
        ),
        Edit(
            id="rd3942-scratch-fields",
            anchor=re.escape(_SCRATCH_FIELD_OLD),
            rationale="add the dedicated concurrent-branch scratch buffer "
                      "fields",
            mode="replace",
            text=_SCRATCH_FIELD_NEW,
            guard=r"ggml_backend_buffer_t concurrent_scratch\s*= nullptr;",
        ),
        # rd3942-cublas-handle-method removed: upstream PR #26574 already
        # widened cublas_handle(int device) to index by (device, stream) --
        # see the _HANDLE_METHOD_OLD/NEW removal note above.
    ),
)

PATCH_GGML_CUDA_CU = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="Honor the active stream, destroy per-stream cuBLAS "
                "handles, replace branch interleaving with dedicated "
                "scratch, and overlap the MoE shared expert (amd-ecosystem "
                "PR #36 / RD39-RD42)",
    edits=(
        Edit(
            id="rd3942-include-unordered-set",
            anchor=re.escape(_INCLUDE_OLD),
            rationale="std::unordered_set is used by the MoE-overlap "
                      "backward-reachability scan",
            mode="replace",
            text=_INCLUDE_NEW,
            guard=r"#include <unordered_set>",
        ),
        Edit(
            id="rd3942-destructor",
            anchor=re.escape(_DTOR_OLD),
            rationale="destroy every (device,stream) cuBLAS handle and "
                      "free the concurrent scratch buffer",
            mode="replace",
            text=_DTOR_NEW,
            guard=r"if \(concurrent_scratch != nullptr\) \{",
        ),
        Edit(
            id="rd39-stream-fallback-join-branch",
            anchor=re.escape(_STREAM_FALLBACK_1_OLD),
            rationale="RD39: nodes outside the concurrent stream_mapping "
                      "(e.g. the routed branch during a shared-expert "
                      "overlap) fall back to the main stream instead of "
                      "asserting",
            mode="replace",
            text=_STREAM_FALLBACK_1_NEW,
            guard=r"region nodes not mapped to a concurrent stream run on the main stream",
        ),
        Edit(
            id="rd39-stream-fallback-post-fusion",
            anchor=re.escape(_STREAM_FALLBACK_2_OLD),
            rationale="RD39: same fallback for the post-fusion resume path",
            mode="replace",
            text=_STREAM_FALLBACK_2_NEW,
            guard=r"try_launch_concurrent_event\(prev_node\);\n\n\s*if \(is_concurrent_event_active\) \{\n\s*auto it = concurrent_event->stream_mapping\.find\(node\);",
        ),
        Edit(
            id="rd3942-concurrent-groups-decl",
            anchor=re.escape(_RANGES_DECL_OLD),
            rationale="declare the per-event branch-node lists consumed by "
                      "the scratch-buffer placement at the end of the "
                      "function",
            mode="replace",
            text=_RANGES_DECL_NEW,
            guard=r"std::vector<std::vector<const ggml_tensor \*>> concurrent_groups;",
        ),
        Edit(
            id="rd3941-drop-interleave-decls",
            anchor=re.escape(_DECLS_OLD),
            rationale="RD41: the in-place interleave loop's index/branch "
                      "counters are no longer needed once branches move to "
                      "dedicated scratch",
            mode="replace",
            text=_DECLS_NEW,
            # A pure deletion has no new text to guard on, so guard on the
            # ABSENCE of the deleted declarations directly above the line
            # that survives them (fixed-width negative lookbehind -- Python
            # requires that for lookbehind, which this literal string is).
            # Reuses _DECLS_OLD itself (minus its last line) so the guard
            # can never drift out of sync with what the edit actually
            # deletes.
            guard=r"(?<!"
                  + re.escape(_DECLS_OLD.rsplit("int total_branch_nodes", 1)[0])
                  + r")int total_branch_nodes = 0;",
        ),
        Edit(
            id="rd3941-scratch-buffer-and-rd42-moe-overlap",
            anchor=_MOE_OVERLAP_ANCHOR,
            rationale="RD41: replace in-place branch interleaving with "
                      "dedicated-scratch placement; RD42: detect and fork "
                      "the MoE shared-expert diamond onto an aux stream, "
                      "then allocate/place the scratch buffer for every "
                      "concurrent group",
            mode="replace",
            text=_MOE_OVERLAP_NEW,
            guard=r"MoE shared-expert overlap: run the shared expert on a separate stream",
        ),
    ),
)

PATCHES = [PATCH_COMMON_CUH, PATCH_GGML_CUDA_CU]
