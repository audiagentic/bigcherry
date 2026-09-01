"""RD73 (FORK-MTP-003, re-scoped): replace the HIP/CUDA graph-cache key
with a stable FNV-1a shape fingerprint, instead of the raw first-node
pointer.

Provenance (group 'rdna-boosts' patches are external backports; the
machine-readable PROVENANCE dict below is cross-checked against
external-sources.toml by tools/tests/test_external_sources.py):

  source:            mrlordcat-rdna-lab
  repo:              https://github.com/MrLordCat/llama.cpp-rdna-lab
  commit:            7f2e7e4a3ebf8e3b5aade75743c267f5ad7df199
  title:             stable graph-cache key (FNV-1a shape fingerprint)
  reviewed:          2026-08-23 (RD69-RD76 fork investigation, re-scoped
                     from the item's original "prebuild verify widths"
                     framing to the real mechanism -- see
                     docs/planning/active/rdna-boost-experiments/RD73.md
                     and RD69-76_FORK_FINDINGS.md).

What it does: `ggml_cuda_graph_get_key()` previously returned
`cgraph->nodes[0]` (the raw pointer of the first graph node) as the
`std::unordered_map<const void *, ...> cuda_graphs` key
(common.cuh:1491). That pointer is allocation-dependent, not
shape-dependent: a fresh allocation for an otherwise-identical
recurring shape (e.g. repeated speculative-verify batches) gets a new
pointer and therefore a cold cache miss almost every time, even though
`ggml_cuda_graph_update_required()` would have found the shape
unchanged and reused the graph. This replaces the key with a 64-bit
FNV-1a fingerprint over the node count plus the first and last nodes'
op/name/ne[], which is stable across allocations for the same
recurring shape.

Collision safety: this key is only a bucket selector, not a substitute
for the existing correctness check. `ggml_cuda_graph_update_required()`
(immediately below the key lookup) still compares `node_props.size()`
against `cgraph->n_nodes` and memcmp()s every node's op/src
pointers/ne/nb before treating a cached graph as reusable -- a
fingerprint collision between two different shapes costs one extra
recapture on that key, it cannot cause stale-graph reuse.

Upstream-fork measured (3.8k-node Qwen3.6-27B graph, repeated
speculative-verify batches): verify ubatch sync 150ms -> 57ms. Not
merged upstream; 34-line one-file change (ggml/src/ggml-cuda/
ggml-cuda.cu) in the fork, ported here as a single anchor replace.

Porting notes:
  - BigCherry's HIP build compiles the generic ggml-cuda source tree
    (ggml/src/ggml-cuda/ggml-cuda.cu) via ggml-hip's #include shim, so
    this is the correct target file -- confirmed by grepping the built
    tree for ggml_cuda_graph_get_key(), which resolves only here.
    Vulkan has a separate, unrelated graph-cache subsystem and is out
    of scope for this patch (the item was originally filed
    Vulkan-primary before the fork investigation found the actual
    mechanism lives in ggml-cuda).
  - No hunk collision with patch 1231 (HI14 graph-lifecycle evidence,
    which instruments cudaStreamBeginCapture/EndCapture/
    cudaGraphInstantiate/cudaGraphLaunch call sites) or patch 1217
    (RD44, which edits ggml_backend_cuda_graph_optimize): both anchors
    verified against vendor/llama.cpp to be disjoint text regions from
    ggml_cuda_graph_get_key().
  - Deliberately narrow: hashes only the node count and the first/last
    node's op/name/ne[], matching the fork's own cheap coarse
    fingerprint rather than hashing every node (which would make the
    key computation itself a per-dispatch cost comparable to the
    per-node comparison it is meant to short-circuit).
  - Ported byte-for-byte from the real fork diff (fetched via `gh api`
    to verify, not just the review's paraphrase), after an external
    review (dev-gpt-agent, 2026-08-23) caught a real bug in an earlier
    draft of this port: that draft hashed `sizeof(node->name)` (the
    whole fixed 64-byte name[] buffer) instead of
    `strnlen(node->name, GGML_MAX_NAME)`. ggml_set_name()/
    ggml_format_name() (ggml.c) write the visible name and its
    terminating NUL but never clear the rest of the buffer, so two
    tensors with the same logical name can carry different stale tail
    bytes there -- hashing the whole buffer would have reintroduced
    exactly the key instability this patch exists to remove. Also
    fixed to match the fork: n_nodes/op normalized through int32_t
    before hashing, and the fork's own (non-standard) FNV-1a offset
    basis literal kept as-is rather than swapped for the canonical
    constant, to avoid an unnecessary causal variable versus the
    fork's own measured result.

Hardware status: UNVALIDATED on BigCherry hardware as of porting --
STATE stays "untested" pending a real-HIP causal comparison (control
vs this patch alone) on a repeated-shape graph workload, per the plan
item's own acceptance criteria (no stale-graph/output errors, and a
measured steady-state verify-latency improvement).

Isolation and promotion (first-sweep policy, matching existing RD
items): GROUP 'rdna-boosts' + STATE 'untested' keeps this OUT of the
production 'framework' and 'validated-enhancements' patch-sets until
real-hardware validation lands.
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

PROVENANCE = {
    "source-id": "mrlordcat-rdna-lab",
    "plan-item": "RD73",
    "fork-commit": "7f2e7e4a3ebf8e3b5aade75743c267f5ad7df199",
    "fork-commit-title": "stable graph-cache key (FNV-1a shape fingerprint)",
    "snapshot-head": "7f2e7e4a3ebf8e3b5aade75743c267f5ad7df199",
    "snapshot-base": "7f2e7e4a3ebf8e3b5aade75743c267f5ad7df199",
    "adaptations": [
        "Mechanism ported byte-for-byte (fingerprint fields, int32_t "
        "normalization, strnlen-bounded name hashing, both FNV-1a "
        "constants including the fork's non-standard offset basis "
        "literal) after an external review (dev-gpt-agent, 2026-08-23) "
        "caught an earlier draft of this port hashing the full "
        "fixed-size name[] buffer instead of strnlen(name, "
        "GGML_MAX_NAME) -- verified against the fork's real commit diff "
        "via `gh api`, not just the review's paraphrase. Comments "
        "rewritten to explain BigCherry's own collision-safety argument "
        "against ggml_cuda_graph_update_required()'s existing per-node "
        "memcmp, why the full-buffer hash was wrong (ggml_set_name/"
        "ggml_format_name never clear name[]'s tail bytes), and which "
        "graph-cache subsystem (ggml-cuda, not Vulkan) this targets.",
    ],
}

_KEY_OLD = """static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    return cgraph->nodes[0];
}"""

_KEY_NEW = """static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    // bigcherry (RD73): stable shape-fingerprint key, replacing the raw
    // first-node pointer. An allocator can hand a fresh address to an
    // otherwise-identical recurring shape (e.g. repeated speculative-verify
    // batches), so the old pointer key almost never hit a warm cache slot.
    // Ported byte-for-byte from MrLordCat/llama.cpp-rdna-lab commit
    // 7f2e7e4a (see config/external-sources.toml) -- including its
    // non-standard FNV-1a offset basis literal below, kept as-is rather
    // than swapped for the canonical 0xcbf29ce484222325 to avoid an
    // unnecessary causal variable versus the fork's own measured result.
    // A fingerprint collision is safe: ggml_cuda_graph_update_required()
    // below still compares graph size and memcmp()s every node's op/src
    // pointers/ne/nb before reusing a cached instance, so a false-positive
    // key match only costs an extra recapture, never stale reuse. Hashing
    // only strnlen(name, GGML_MAX_NAME) bytes (not the full fixed buffer)
    // matters: ggml_set_name()/ggml_format_name() (ggml.c) write the name
    // and its NUL but never clear the rest of the fixed-size name[]
    // buffer, so two tensors with the same logical name can carry
    // different stale tail bytes there -- hashing the whole buffer would
    // reintroduce the exact key instability this patch exists to remove.

    // bigcherry: HI83 activation-evidence instrumentation, not part of
    // the ported fork change. Once-per-process, opt-in via
    // BIGCHERRY_PATCH_TRACE (same pattern as RD08/RD12's markers).
    // GGML_LOG_WARN, not INFO: llama-bench's own --verbose gate on top
    // of ggml's log level filters both by default, but the trace-probe
    // path this marker is read through always requests --verbose (VA21
    // real-hardware finding) -- WARN still matches HI14's own convention
    // for opt-in activation evidence at this project's default verbosity.
    if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
        static std::atomic_flag bigcherry_rd73_logged = ATOMIC_FLAG_INIT;
        if (!bigcherry_rd73_logged.test_and_set(std::memory_order_relaxed)) {
            GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key\\n");
        }
    }

    uint64_t h = 1469598103934665603ULL; // FNV-1a offset basis (fork's literal)
    auto mix = [&h](const void * data, size_t n) {
        const unsigned char * p = (const unsigned char *) data;
        for (size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 1099511628211ULL; // FNV-1a 64-bit prime
        }
    };

    const int32_t n_nodes = cgraph->n_nodes;
    mix(&n_nodes, sizeof(n_nodes));
    if (n_nodes > 0) {
        const ggml_tensor * first = cgraph->nodes[0];
        const int32_t op_f = (int32_t) first->op;
        mix(&op_f, sizeof(op_f));
        mix(first->name, strnlen(first->name, GGML_MAX_NAME));
        mix(first->ne, sizeof(first->ne));

        const ggml_tensor * last = cgraph->nodes[n_nodes - 1];
        const int32_t op_l = (int32_t) last->op;
        mix(&op_l, sizeof(op_l));
        mix(last->name, strnlen(last->name, GGML_MAX_NAME));
        mix(last->ne, sizeof(last->ne));
    }

    return (const void *) (uintptr_t) h;
}"""


PATCH = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="Replace the CUDA/HIP graph-cache key with a stable "
                "FNV-1a shape fingerprint instead of the raw first-node "
                "pointer (mrlordcat-rdna-lab 7f2e7e4a / RD73)",
    edits=(
        Edit(
            id="rd73-stable-graph-cache-key",
            anchor=re.escape(_KEY_OLD),
            rationale="ggml_cuda_graph_get_key(): key on a shape "
                      "fingerprint instead of the allocation-dependent "
                      "first-node pointer, so recurring shapes (e.g. "
                      "speculative-verify batches) hit a warm cached graph",
            mode="replace",
            text=_KEY_NEW,
            guard=r"FNV-1a offset basis \(fork's literal\)",
        ),
    ),
)

# VA06: RD73's own resource-cost claim (increased graph-cache key
# cardinality, since a stable fingerprint no longer collapses distinct
# shapes onto a shared allocation-dependent pointer -- VA06's real
# characterization measured 386 -> 651 entries) needs a real, repeatable
# machine-readable evidence producer, not the earlier one-off temporary
# instrumentation. Instruments the SAME real insertion site
# (cuda_graph()'s cuda_graphs.emplace() in common.cuh) characterized
# then, opt-in and machine-parseable this time.
_RESOURCE_OLD = """        auto it = cuda_graphs.find(first_node_ptr);
        if (it == cuda_graphs.end()) {
            it = cuda_graphs.emplace(first_node_ptr, std::make_unique<ggml_cuda_graph>()).first;
        }"""

_RESOURCE_NEW = """        auto it = cuda_graphs.find(first_node_ptr);
        if (it == cuda_graphs.end()) {
            it = cuda_graphs.emplace(first_node_ptr, std::make_unique<ggml_cuda_graph>()).first;
            // bigcherry (RD73/VA06): opt-in, machine-readable graph-cache
            // resource-cost telemetry -- NOT part of the ported fork
            // change. Emits on every new-entry insertion (not once) so an
            // offline reducer can take the peak over a real workload;
            // gated separately from BIGCHERRY_PATCH_TRACE (the activation
            // marker above) since this is a resource measurement, not an
            // activation proof, and a caller may want either independently.
            if (getenv("BIGCHERRY_RD73_RESOURCE_TRACE") != nullptr) {
                GGML_LOG_WARN("BIGCHERRY_RD73_RESOURCE graph_cache_entries=%zu\\n", cuda_graphs.size());
            }
        }"""

RESOURCE_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/common.cuh",
    description="Opt-in, machine-readable graph-cache resource-cost "
                "telemetry at the real cuda_graphs insertion site (RD73 "
                "/ VA06)",
    edits=(
        Edit(
            id="rd73-resource-telemetry",
            anchor=re.escape(_RESOURCE_OLD),
            rationale="cuda_graph(): emit the real cuda_graphs.size() on "
                      "every new-entry insertion, opt-in via "
                      "BIGCHERRY_RD73_RESOURCE_TRACE, so a real executor "
                      "can measure the contract's resource_limits.graph_"
                      "cache_entries claim without one-off temporary "
                      "instrumentation",
            mode="replace",
            text=_RESOURCE_NEW,
            guard=r"BIGCHERRY_RD73_RESOURCE_TRACE",
        ),
    ),
)

PATCHES = [PATCH, RESOURCE_PATCH]
