"""1316 (QFP15): per-node output hash trace for bisecting run-to-run nondeterminism to one graph node.

Companion to 1315 (which says whether the target or the draft diverges first). With BIGCHERRY_NODE_HASH="from:count",
every llama_context counts its graph_compute calls; for calls from..from+count-1 a scheduler eval callback copies each
computed node's output to host and logs `BIGCHERRY_NODE_HASH ctx=<context> g=<graph index> i=<node seq> name op ne
h=<FNV-1a of the bytes>`. Run the same build twice on the same prompt, diff the streams per context: the first
differing line is the earliest node whose output is not bit-reproducible. Graph indices align across runs up to the
first divergence (same prompt chunks, same decode steps). Installed only when the caller set no cb_eval of its own.
The eval callback forces per-node synchronization and disables CUDA/HIP graph replay, so a race that needs overlap may
not reproduce under it; then narrow the window with from:count and compare against 1315's unperturbed trace.
Diagnostic only; never in a production recipe.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_INCLUDE_ANCHOR = "#include <unordered_map>\n"
_INCLUDE = r"""#include <atomic>   // bigcherry 1316
#include <cstdio>   // bigcherry 1316: std::sscanf
#include <cstdlib>  // bigcherry 1316: std::getenv
#include <mutex>    // bigcherry 1316
#include <vector>   // bigcherry 1316

// bigcherry 1316: BIGCHERRY_NODE_HASH="from:count" per-node output hashes for a window of graph_compute calls.
struct bc_node_hash_state {
    std::mutex mu;
    std::unordered_map<const void *, long> graph_idx;  // per llama_context: graph_compute calls so far
    std::unordered_map<const void *, bool> active;     // per llama_context: current graph inside the window
    long from = -1;
    long count = 0;
    long node_seq = 0;
};

static bc_node_hash_state & bc_node_hash() {
    static bc_node_hash_state s;
    static bool parsed = false;
    if (!parsed) {
        parsed = true;
        if (const char * env = std::getenv("BIGCHERRY_NODE_HASH")) {
            if (std::sscanf(env, "%ld:%ld", &s.from, &s.count) != 2 || s.from < 0 || s.count <= 0) {
                throw std::runtime_error("BIGCHERRY_NODE_HASH must be from:count with from >= 0 and count > 0");
            }
        }
    }
    return s;
}

static bool bc_node_hash_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    bc_node_hash_state & s = bc_node_hash();
    bool on = false;
    long g = -1;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        on = s.active[user_data];
        g = s.graph_idx[user_data] - 1;
    }
    if (ask || !on) {
        return on;  // only request node data inside the window
    }
    const size_t n = ggml_nbytes(t);
    std::vector<unsigned char> buf(n);
    ggml_backend_tensor_get(t, buf.data(), 0, n);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ buf[i]) * 1099511628211ull;
    }
    long seq = 0;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        seq = s.node_seq++;
    }
    LLAMA_LOG_WARN("BIGCHERRY_NODE_HASH ctx=%p g=%ld i=%ld name=%s op=%s ne=%lld,%lld,%lld,%lld h=%016llx\n",
        user_data, g, seq, t->name, ggml_op_desc(t), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
        (long long) t->ne[3], (unsigned long long) h);
    return true;
}
"""

_CTOR_ANCHOR = "    cparams.cb_eval_user_data = params.cb_eval_user_data;\n"
_CTOR = r"""    if (cparams.cb_eval == nullptr && bc_node_hash().from >= 0) {  // bigcherry 1316
        cparams.cb_eval           = bc_node_hash_cb;
        cparams.cb_eval_user_data = this;
        LLAMA_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1316_node_hash_trace ctx=%p from=%ld count=%ld\n",
                       (void *) this, bc_node_hash().from, bc_node_hash().count);
    }
"""

_COMPUTE_ANCHOR = "    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);\n"
_COMPUTE = r"""    if (cparams.cb_eval == bc_node_hash_cb) {  // bigcherry 1316: advance this context's graph index
        bc_node_hash_state & s = bc_node_hash();
        std::lock_guard<std::mutex> lock(s.mu);
        const long g = s.graph_idx[this]++;
        s.active[this] = g >= s.from && g < s.from + s.count;
    }
"""

PATCHES = [
    FilePatch(
        path="src/llama-context.cpp",
        description="1316: BIGCHERRY_NODE_HASH per-node output hash trace (diagnostic)",
        language="none",
        edits=(
            Edit(
                id="node-hash-helpers",
                anchor=re.escape(_INCLUDE_ANCHOR),
                mode="insert_after",
                text=_INCLUDE,
                guard=r"static bool bc_node_hash_cb\(",
                rationale="Last standard include of llama-context.cpp; helpers are file-static.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="node-hash-ctor",
                anchor=re.escape(_CTOR_ANCHOR),
                mode="insert_after",
                text=_CTOR,
                guard=r"cparams.cb_eval           = bc_node_hash_cb;",
                rationale="Context constructor, right after the caller's cb_eval is copied into cparams.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="node-hash-compute",
                anchor=re.escape(_COMPUTE_ANCHOR),
                mode="insert_before",
                text=_COMPUTE,
                guard=r"bigcherry 1316: advance this context's graph index",
                rationale="llama_context::graph_compute, immediately before the scheduler runs the graph.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_NODE_HASH', '<from>:<count>', 'unset',
           'diagnostic: hash graph node outputs [from, from+count) to locate nondeterminism'),
)
