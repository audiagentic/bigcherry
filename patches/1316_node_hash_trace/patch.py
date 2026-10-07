"""1316 (QFP15): per-node output hash trace for bisecting run-to-run nondeterminism to one graph node.

Companion to 1315 (which says whether the target or the draft diverges first). With BIGCHERRY_NODE_HASH="from:count",
every llama_context counts its graph_compute calls; for calls from..from+count-1 a scheduler eval callback copies each
computed node's output to host and logs `BIGCHERRY_NODE_HASH ctx=<context> g=<graph index> i=<node seq> name op ne
h=<FNV-1a of the bytes>`. Run the same build twice on the same prompt, diff the streams per context: the first
differing line is the earliest node whose output is not bit-reproducible. Graph indices align across runs up to the
first divergence (same prompt chunks, same decode steps). Installed only when the caller set no cb_eval of its own.
On ordinary backends the eval callback forces per-node synchronization and disables CUDA/HIP graph replay. Meta scheduler
splits are kept intact and synchronized once because Meta's internal subgraph/reduction walk cannot accept arbitrary node
sub-ranges; Meta-owned nodes are then reported as `skip=meta` and are not read. Diagnostic only; never in a production recipe.
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

static bool bc_node_hash_is_meta_tensor(const struct ggml_tensor * t) {
    if (t == nullptr) {
        return false;
    }
    ggml_backend_buffer_t buffer = t->view_src != nullptr ? t->view_src->buffer : t->buffer;
    if (buffer == nullptr) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buffer);
    if (buft == nullptr) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
    return device != nullptr && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_META;
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
    // A Meta buffer may represent a split/partial logical tensor. Its generic get path can abort for
    // shapes/offsets that are not representable as one flat host read. This diagnostic must never
    // turn observation into a process abort, so skip Meta-owned nodes and keep hashing simple buffers.
    if (bc_node_hash_is_meta_tensor(t)) {
        long seq = 0;
        {
            std::lock_guard<std::mutex> lock(s.mu);
            seq = s.node_seq++;
        }
        LLAMA_LOG_WARN("BIGCHERRY_NODE_HASH ctx=%p g=%ld i=%ld name=%s op=%s ne=%lld,%lld,%lld,%lld skip=meta\n",
            user_data, g, seq, t->name, ggml_op_desc(t), (long long) t->ne[0], (long long) t->ne[1],
            (long long) t->ne[2], (long long) t->ne[3]);
        return true;
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

_HEADER_EVAL_ANCHOR = """    GGML_API void                 ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);
"""
_HEADER_EVAL_NEW = """    GGML_API void                 ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);
    // BigCherry 1316 diagnostic: keep Meta scheduler splits atomic while the node-hash callback observes them.
    GGML_API void                 ggml_backend_sched_set_eval_callback_meta_atomic(ggml_backend_sched_t sched, bool enabled);
"""

_SCHED_STATE_ANCHOR = """    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;
"""
_SCHED_STATE_NEW = """    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;
    bool callback_eval_meta_atomic; // BigCherry 1316: only the node-hash diagnostic enables this.
"""

_SCHED_SETTER_ANCHOR = """void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}
"""
_SCHED_SETTER_NEW = """void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

void ggml_backend_sched_set_eval_callback_meta_atomic(ggml_backend_sched_t sched, bool enabled) {
    GGML_ASSERT(sched);
    sched->callback_eval_meta_atomic = enabled;
}
"""

_CTX_EVAL_ANCHOR = """        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
"""
_CTX_EVAL_NEW = """        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
        ggml_backend_sched_set_eval_callback_meta_atomic(sched.get(), cparams.cb_eval == bc_node_hash_cb);
"""

_SCHED_EVAL_ANCHOR = """        } else {
            // similar to ggml_backend_compare_graph_backend
"""
_SCHED_EVAL_NEW = """        } else if (sched->callback_eval_meta_atomic &&
                ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_META) {
            // BigCherry 1316: Meta owns its own subgraph partition/reduction walk. Never feed it the node-range
            // graph views used by the generic eval-callback path: those views are not valid Meta graph boundaries.
            // Compute the scheduler split exactly as without a callback, then report its nodes. 1316's callback
            // deliberately reports Meta-buffer nodes as skip=meta, so no unsafe logical-tensor read is introduced.
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
            ggml_backend_synchronize(split_backend);
            for (int j = 0; j < split->graph.n_nodes; ++j) {
                struct ggml_tensor * t = split->graph.nodes[j];
                if (sched->callback_eval(t, true, sched->callback_eval_user_data) &&
                        !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
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
            Edit(
                id="node-hash-meta-atomic-enable",
                anchor=re.escape(_CTX_EVAL_ANCHOR),
                mode="replace",
                text=_CTX_EVAL_NEW,
                guard=r"set_eval_callback_meta_atomic\(sched\.get\(\), cparams\.cb_eval == bc_node_hash_cb\)",
                rationale="Enable Meta split atomicity only when 1316's own node-hash callback is installed.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="ggml/include/ggml-backend.h",
        description="1316: opt-in scheduler flag for Meta-atomic eval callbacks",
        language="none",
        edits=(
            Edit(
                id="node-hash-meta-atomic-api",
                anchor=re.escape(_HEADER_EVAL_ANCHOR),
                mode="replace",
                text=_HEADER_EVAL_NEW,
                guard=r"ggml_backend_sched_set_eval_callback_meta_atomic",
                rationale="Expose a diagnostic-only scheduler switch so 1316 does not alter unrelated eval callbacks.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-backend.cpp",
        description="1316: keep Meta scheduler splits intact only for the node-hash eval callback",
        language="none",
        edits=(
            Edit(
                id="node-hash-meta-atomic-state",
                anchor=re.escape(_SCHED_STATE_ANCHOR),
                mode="replace",
                text=_SCHED_STATE_NEW,
                guard=r"callback_eval_meta_atomic",
                rationale="Store the explicit 1316-only Meta stepping mode beside the eval callback state.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="node-hash-meta-atomic-setter",
                anchor=re.escape(_SCHED_SETTER_ANCHOR),
                mode="replace",
                text=_SCHED_SETTER_NEW,
                guard=r"void ggml_backend_sched_set_eval_callback_meta_atomic",
                rationale="Set the 1316-only Meta stepping mode without changing normal callback registration semantics.",
                expect_matches=1,
                max_span_lines=6,
            ),
            Edit(
                id="node-hash-meta-split-atomic",
                anchor=re.escape(_SCHED_EVAL_ANCHOR),
                mode="replace",
                text=_SCHED_EVAL_NEW,
                guard=r"BigCherry 1316: Meta owns its own subgraph partition/reduction walk",
                rationale="Meta graph_compute requires the complete scheduler split; generic per-node graph views violate its subgraph walk.",
                expect_matches=1,
                max_span_lines=3,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_NODE_HASH', '<from>:<count>', 'unset',
           'diagnostic: hash graph node outputs [from, from+count) to locate nondeterminism'),
)
