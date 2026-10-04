"""1331 (QFP17 chunked scratch): log the tensors that are live at each compute buffer's allocation peak.

1329 ranks the largest tensors in a reserved graph, but the compute buffer size is set by the PEAK of simultaneously
live tensors, not by the largest ones: removing a large short-lived tensor (1330's second QSA mask) left the ub1024
buffer unchanged at 1560.72 MiB. With BIGCHERRY_ALLOC_PEAK=N, ggml_gallocr_alloc_graph_impl tracks, per buffer id, the
bytes currently allocated (allocate/free of own tensors; in-place reuse transfers ownership without changing the
total), and at every new peak snapshots the live set. After the graph it logs, per buffer: the peak live bytes, the
node being allocated at the peak, the N largest live tensors at that moment and live bytes per op. Fragmentation
is not modelled (the dynamic allocator's chunk size can exceed the live peak); compare with the sched_reserve size.
Diagnostic only; no behaviour change without the variable.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_A_HELPERS = "static void ggml_gallocr_allocate_node(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id) {\n"
_N_HELPERS = r"""// bigcherry 1331: live bytes at each buffer's allocation peak (BIGCHERRY_ALLOC_PEAK=N)
enum { BC_PEAK_BUFS = 16, BC_PEAK_TOP = 64 };
static struct {
    int on;
    int top;
    size_t cur[BC_PEAK_BUFS];
    size_t peak[BC_PEAK_BUFS];
    const struct ggml_tensor * at[BC_PEAK_BUFS];
    const struct ggml_tensor * live[BC_PEAK_BUFS][BC_PEAK_TOP];
    size_t live_size[BC_PEAK_BUFS][BC_PEAK_TOP];
    int n_live[BC_PEAK_BUFS];
    const char * op_name[BC_PEAK_BUFS][32];
    size_t op_bytes[BC_PEAK_BUFS][32];
    int n_ops[BC_PEAK_BUFS];
} bc_peak;

static void bc_peak_snapshot(ggml_gallocr_t galloc, int b, const struct ggml_tensor * at) {
    bc_peak.peak[b] = bc_peak.cur[b];
    bc_peak.at[b] = at;
    bc_peak.n_live[b] = 0;
    bc_peak.n_ops[b] = 0;
    for (size_t i = 0; i < galloc->hash_set.size; i++) {
        if (!ggml_bitset_get(galloc->hash_set.used, i)) {
            continue;
        }
        const struct hash_node * hn = &galloc->hash_values[i];
        const struct ggml_tensor * t = galloc->hash_set.keys[i];
        if (!hn->allocated || hn->buffer_id != b || t == NULL || ggml_impl_is_view(t)) {
            continue;
        }
        const size_t sz = ggml_backend_buft_get_alloc_size(galloc->bufts[b], (struct ggml_tensor *) t);
        const char * od = ggml_op_desc(t);
        int j = 0;
        while (j < bc_peak.n_ops[b] && strcmp(bc_peak.op_name[b][j], od) != 0) {
            j++;
        }
        if (j == bc_peak.n_ops[b] && j < 32) {
            bc_peak.op_name[b][j] = od;
            bc_peak.op_bytes[b][j] = 0;
            bc_peak.n_ops[b]++;
        }
        if (j < 32) {
            bc_peak.op_bytes[b][j] += sz;
        }
        int pos = bc_peak.n_live[b];  // keep the largest (small sorted array)
        while (pos > 0 && bc_peak.live_size[b][pos - 1] < sz) {
            pos--;
        }
        if (pos < bc_peak.top) {
            const int last = bc_peak.n_live[b] < bc_peak.top ? bc_peak.n_live[b] : bc_peak.top - 1;
            for (int m = last; m > pos; m--) {
                bc_peak.live[b][m] = bc_peak.live[b][m - 1];
                bc_peak.live_size[b][m] = bc_peak.live_size[b][m - 1];
            }
            bc_peak.live[b][pos] = t;
            bc_peak.live_size[b][pos] = sz;
            if (bc_peak.n_live[b] < bc_peak.top) {
                bc_peak.n_live[b]++;
            }
        }
    }
}

static void bc_peak_alloc(ggml_gallocr_t galloc, int b, const struct ggml_tensor * node, size_t size) {
    if (!bc_peak.on || b < 0 || b >= BC_PEAK_BUFS) {
        return;
    }
    bc_peak.cur[b] += size;
    if (bc_peak.cur[b] > bc_peak.peak[b]) {
        bc_peak_snapshot(galloc, b, node);
    }
}

static void bc_peak_free(int b, size_t size) {
    if (bc_peak.on && b >= 0 && b < BC_PEAK_BUFS) {
        bc_peak.cur[b] -= size;
    }
}

static void bc_peak_begin(void) {
    const char * env = getenv("BIGCHERRY_ALLOC_PEAK");
    const int k = env ? atoi(env) : 0;
    memset(&bc_peak, 0, sizeof(bc_peak));
    bc_peak.on = k > 0;
    bc_peak.top = k < BC_PEAK_TOP ? k : BC_PEAK_TOP;
}

static void bc_peak_report(const struct ggml_cgraph * graph) {
    if (!bc_peak.on) {
        return;
    }
    for (int b = 0; b < BC_PEAK_BUFS; b++) {
        if (bc_peak.peak[b] == 0) {
            continue;
        }
        const struct ggml_tensor * at = bc_peak.at[b];
        GGML_LOG_WARN("BIGCHERRY_ALLOC_PEAK graph n_nodes=%d buf=%d peak_live %.1f MiB at op=%s name=%s\n", graph->n_nodes, b,
            (double) bc_peak.peak[b] / (1024.0 * 1024.0), at ? ggml_op_desc(at) : "-", at ? at->name : "-");
        for (int m = 0; m < bc_peak.n_live[b]; m++) {
            const struct ggml_tensor * t = bc_peak.live[b][m];
            GGML_LOG_WARN("BIGCHERRY_ALLOC_PEAK buf=%d %2d %9.2f MiB op=%s name=%s ne=%lld,%lld,%lld,%lld type=%s\n", b, m,
                (double) bc_peak.live_size[b][m] / (1024.0 * 1024.0), ggml_op_desc(t), t->name, (long long) t->ne[0],
                (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], ggml_type_name(t->type));
        }
        for (int a = 0; a < bc_peak.n_ops[b]; a++) {  // per-op live bytes at the peak, largest first
            int best = a;
            for (int c = a + 1; c < bc_peak.n_ops[b]; c++) {
                if (bc_peak.op_bytes[b][c] > bc_peak.op_bytes[b][best]) {
                    best = c;
                }
            }
            const char * tn = bc_peak.op_name[b][a]; bc_peak.op_name[b][a] = bc_peak.op_name[b][best]; bc_peak.op_name[b][best] = tn;
            const size_t tb = bc_peak.op_bytes[b][a]; bc_peak.op_bytes[b][a] = bc_peak.op_bytes[b][best]; bc_peak.op_bytes[b][best] = tb;
            if (a < 12) {
                GGML_LOG_WARN("BIGCHERRY_ALLOC_PEAK buf=%d op %-16s %9.1f MiB\n", b, bc_peak.op_name[b][a],
                    (double) bc_peak.op_bytes[b][a] / (1024.0 * 1024.0));
            }
        }
    }
}

"""

_A_ALLOC = ("        hn->buffer_id = buffer_id;\n"
            "        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, node);\n"
            "    }\n")
_N_ALLOC = ("        hn->buffer_id = buffer_id;\n"
            "        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, node);\n"
            "        bc_peak_alloc(galloc, buffer_id, node, size);  // bigcherry 1331\n"
            "    }\n")

_A_FREE = ("    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);\n"
           "    hn->allocated = false;\n")
_N_FREE = ("    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);\n"
           "    bc_peak_free(buffer_id, size);  // bigcherry 1331\n"
           "    hn->allocated = false;\n")

_A_BEGIN = ("    // clear hash tables\n"
            "    ggml_hash_set_reset(&galloc->hash_set);\n"
            "    memset(galloc->hash_values, 0, sizeof(struct hash_node) * galloc->hash_set.size);\n"
            "\n"
            "    // allocate leafs\n")
_N_BEGIN = ("    // clear hash tables\n"
            "    ggml_hash_set_reset(&galloc->hash_set);\n"
            "    memset(galloc->hash_values, 0, sizeof(struct hash_node) * galloc->hash_set.size);\n"
            "    bc_peak_begin();  // bigcherry 1331\n"
            "\n"
            "    // allocate leafs\n")

_A_END = ("                else if (p_hn->allocated) {\n"
          "                    ggml_gallocr_free_node(galloc, parent);\n"
          "                }\n"
          "            }\n"
          "            AT_PRINTF(\"\\n\");\n"
          "        }\n"
          "    }\n"
          "}\n")
_N_END = ("                else if (p_hn->allocated) {\n"
          "                    ggml_gallocr_free_node(galloc, parent);\n"
          "                }\n"
          "            }\n"
          "            AT_PRINTF(\"\\n\");\n"
          "        }\n"
          "    }\n"
          "    bc_peak_report(graph);  // bigcherry 1331\n"
          "}\n")

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-alloc.c",
        description="1331: BIGCHERRY_ALLOC_PEAK live tensors at each compute buffer's allocation peak",
        language="none",
        edits=(
            Edit(id="alloc-peak-helpers", anchor=re.escape(_A_HELPERS), mode="insert_before", text=_N_HELPERS,
                 guard=r"bigcherry 1331: live bytes at each buffer", rationale="Before ggml_gallocr_allocate_node.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="alloc-peak-alloc", anchor=re.escape(_A_ALLOC), mode="replace", text=_N_ALLOC,
                 guard=r"bc_peak_alloc\(galloc, buffer_id, node, size\);", rationale="ggml_gallocr_allocate_node fresh allocation.",
                 expect_matches=1, max_span_lines=4),
            Edit(id="alloc-peak-free", anchor=re.escape(_A_FREE), mode="replace", text=_N_FREE,
                 guard=r"bc_peak_free\(buffer_id, size\);", rationale="ggml_gallocr_free_node.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="alloc-peak-begin", anchor=re.escape(_A_BEGIN), mode="replace", text=_N_BEGIN,
                 guard=r"bc_peak_begin\(\);", rationale="Start of ggml_gallocr_alloc_graph_impl.",
                 expect_matches=1, max_span_lines=6),
            Edit(id="alloc-peak-report", anchor=re.escape(_A_END), mode="replace", text=_N_END,
                 guard=r"bc_peak_report\(graph\);", rationale="End of ggml_gallocr_alloc_graph_impl.",
                 expect_matches=1, max_span_lines=9),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_ALLOC_PEAK", "<n>", "0 (off)",
           "diagnostic: per compute buffer, log peak live bytes and the n largest tensors live at the peak"),
)
