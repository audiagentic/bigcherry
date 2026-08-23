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
        "Mechanism ported as-is (fingerprint fields, FNV-1a constants); "
        "comments rewritten to explain BigCherry's own collision-safety "
        "argument against ggml_cuda_graph_update_required()'s existing "
        "per-node memcmp, and to record which graph-cache subsystem "
        "(ggml-cuda, not Vulkan) this targets and why.",
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
    // Ported from MrLordCat/llama.cpp-rdna-lab commit 7f2e7e4a (see
    // config/external-sources.toml). A fingerprint collision is safe:
    // ggml_cuda_graph_update_required() below still compares graph size and
    // memcmp()s every node's op/src pointers/ne/nb before reusing a cached
    // instance, so a false-positive key match only costs an extra
    // recapture, never stale reuse.
    uint64_t hash = 0xcbf29ce484222325ULL; // FNV-1a 64-bit offset basis
    auto fnv1a = [&hash](const void * data, size_t len) {
        const unsigned char * bytes = (const unsigned char *) data;
        for (size_t i = 0; i < len; ++i) {
            hash ^= bytes[i];
            hash *= 0x100000001b3ULL; // FNV-1a 64-bit prime
        }
    };
    auto hash_node = [&fnv1a](const ggml_tensor * node) {
        fnv1a(&node->op, sizeof(node->op));
        fnv1a(node->name, sizeof(node->name));
        fnv1a(node->ne, sizeof(node->ne));
    };
    fnv1a(&cgraph->n_nodes, sizeof(cgraph->n_nodes));
    if (cgraph->n_nodes > 0) {
        hash_node(cgraph->nodes[0]);
        hash_node(cgraph->nodes[cgraph->n_nodes - 1]);
    }
    return (const void *) (uintptr_t) hash;
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
            guard=r"FNV-1a 64-bit offset basis",
        ),
    ),
)

PATCHES = [PATCH]
