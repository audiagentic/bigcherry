"""1329 (QFP17 chunked scratch): log the largest compute-buffer tensors at graph reserve.

Larger -ub (1024+) needs ~1.3 GiB more per XTX and ~2 GiB more on the R9700 at 240K f16 on profile v6 (fit probes,
2026-10-04). To decide whether "chunked scratch" (keeping peak buffers at the ub512 level) is worth building, we need
to know which ops own that growth: MoE intermediates (chunking them loses the weight-reuse that makes a larger ubatch
fast) or attention / QSA indexer / mask buffers that scale with n_kv x n_tokens (chunking those keeps the MoE win).
With BIGCHERRY_ALLOC_TOP=N, ggml_gallocr_reserve_n logs, for every reserved graph, the N largest non-view node tensors
(MiB, op, name, shape, target buffer id) and the summed bytes per op. Sizes are logical tensor sizes (a meta-split
tensor reports its full size), and the allocator may reuse memory between tensors, so this ranks contributors rather
than reproducing the exact peak. Diagnostic only.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_A = ("bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {\n"
      "    return ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);\n"
      "}\n")

_N = r"""// bigcherry 1329: largest compute-buffer tensors per reserved graph (BIGCHERRY_ALLOC_TOP=N)
static void bc_alloc_top_trace(const struct ggml_cgraph * graph, const int * node_buffer_ids) {
    const char * env = getenv("BIGCHERRY_ALLOC_TOP");
    const int k = env ? atoi(env) : 0;
    if (k <= 0) {
        return;
    }
    enum { BC_MAX_TOP = 64, BC_MAX_OPS = 96 };
    int top[BC_MAX_TOP];
    int n_top = 0;
    const int kk = k < BC_MAX_TOP ? k : BC_MAX_TOP;
    const char * op_names[BC_MAX_OPS];
    double op_mib[BC_MAX_OPS];
    int n_ops = 0;
    double total = 0.0;
    for (int i = 0; i < graph->n_nodes; i++) {
        const struct ggml_tensor * t = graph->nodes[i];
        if (t->view_src != NULL) {
            continue;
        }
        const double mib = (double) ggml_nbytes(t) / (1024.0 * 1024.0);
        total += mib;
        const char * od = ggml_op_desc(t);
        int j = 0;
        while (j < n_ops && strcmp(op_names[j], od) != 0) {
            j++;
        }
        if (j == n_ops && n_ops < BC_MAX_OPS) {
            op_names[n_ops] = od;
            op_mib[n_ops] = 0.0;
            n_ops++;
        }
        if (j < n_ops) {
            op_mib[j] += mib;
        }
        // keep the kk largest (insertion into a small sorted array)
        int pos = n_top;
        while (pos > 0 && ggml_nbytes(graph->nodes[top[pos - 1]]) < ggml_nbytes(t)) {
            pos--;
        }
        if (pos < kk) {
            const int last = n_top < kk ? n_top : kk - 1;
            for (int m = last; m > pos; m--) {
                top[m] = top[m - 1];
            }
            top[pos] = i;
            if (n_top < kk) {
                n_top++;
            }
        }
    }
    GGML_LOG_WARN("BIGCHERRY_ALLOC_TOP graph n_nodes=%d non-view total %.1f MiB\n", graph->n_nodes, total);
    for (int m = 0; m < n_top; m++) {
        const struct ggml_tensor * t = graph->nodes[top[m]];
        GGML_LOG_WARN("BIGCHERRY_ALLOC_TOP %2d %9.2f MiB buf=%d op=%s name=%s ne=%lld,%lld,%lld,%lld type=%s\n", m,
            (double) ggml_nbytes(t) / (1024.0 * 1024.0), node_buffer_ids ? node_buffer_ids[top[m]] : 0, ggml_op_desc(t),
            t->name, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], ggml_type_name(t->type));
    }
    for (int a = 0; a < n_ops; a++) {  // per-op sums, largest first (small n: selection)
        int best = a;
        for (int b = a + 1; b < n_ops; b++) {
            if (op_mib[b] > op_mib[best]) {
                best = b;
            }
        }
        const char * tn = op_names[a]; op_names[a] = op_names[best]; op_names[best] = tn;
        const double tm = op_mib[a]; op_mib[a] = op_mib[best]; op_mib[best] = tm;
        if (a < 12) {
            GGML_LOG_WARN("BIGCHERRY_ALLOC_TOP op %-16s %9.1f MiB\n", op_names[a], op_mib[a]);
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    bc_alloc_top_trace(graph, node_buffer_ids);  // bigcherry 1329
    return ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);
}
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-alloc.c",
        description="1329: BIGCHERRY_ALLOC_TOP largest compute-buffer tensors per reserved graph",
        language="none",
        edits=(
            Edit(id="alloc-top", anchor=re.escape(_A), mode="replace", text=_N,
                 guard=r"bigcherry 1329: largest compute-buffer tensors", rationale="Public reserve entry point.",
                 expect_matches=1, max_span_lines=4),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_ALLOC_TOP', '<n>', '0 (off)',
           'diagnostic: log the n largest compute-buffer tensors per reserved graph'),
)
