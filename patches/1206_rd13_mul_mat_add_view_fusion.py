"""RD13: fuse mul_mat + add through a view (reshape) node.

Provenance (group 'rdna-boosts' patches are external backports; the
machine-readable PROVENANCE dict below is cross-checked against
external-sources.toml by tools/tests/test_external_sources.py):

  source:            stew675-rdna-boosts
  repo:              https://github.com/stew675/llama.cpp
  locator:           rdna-boosts (branch name is a locator only, NOT identity)
  fork commit:       0153d580dbc2caa8f29b55ad8ddc7088b4c457dd
                     (snapshot v2; v1 ledger item 15, 36270950d, is the
                     pre-rebase identity of the SAME logical change,
                     content-identical per git patch-id)
                     "CUDA: fuse mul_mat + add through a view node"
  reviewed snapshot: v2 -- head 9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22
                     on base 4df29be4f4c3673f428170fda944a5b19f743bb8
  plan item:         RD13 (docs/planning/active/rdna-boost-experiments/RD13.md)
  mainline status:   NOT merged into ggml-org/llama.cpp master as of tip
                     6d0549831 (git cherry patch-id check, 2026-08-18)

What it does (performance, graph-level fusion):
  The existing mul_mat + add fusion in ggml_cuda_try_fuse only matched
  the add DIRECTLY after the matmul. SSM models (e.g. qwen35moe) insert
  a RESHAPE view between the output projection and the residual add, so
  the fusion never fired and every layer ran a separate add kernel.
  This accepts one RESHAPE node between the matmul and the add, using
  ggml_can_fuse_subgraph (ggml_can_fuse rejects view nodes in non-last
  positions), verifying the view's src[0] IS the matmul node.
  fused_node_count becomes 3 when the view is present so the fusion
  bookkeeping consumes the right number of graph nodes.

Porting notes:
  - Ported VERBATIM from the fork commit: the entire "mul_mat + add"
    branch of ggml_cuda_try_fuse is replaced (one hunk). The pre-change
    block is byte-identical on the framework-patched base (pin
    4801e3c567d5 + core + upstream-fixes patches): verified 2026-08-19.
  - No framework patch anchors inside this block (the ggml-cuda.cu
    co-tenants 0200/0700/0830/0900/1004 all edit other regions).
  - ggml_can_fuse_subgraph is a pre-existing upstream ggml API -- no new
    symbols.
  - Interaction note (deferred by first-sweep policy): RD17 (MoE top-k
    into down projection) also edits ggml-cuda.cu's dispatch/fusion
    region. They are separate patches against the same base; any
    conflict surfaces only if BOTH are selected, which the isolated
    bench policy avoids.

Isolation and promotion (first-sweep policy, RD review 2026-08-18):
  - GROUP 'rdna-boosts' + STATE 'untested' keeps this OUT of the
    production 'framework' and 'validated-enhancements' patch-sets.
  - The bench needs an SSM/Mamba-family model (the view pattern is what
    makes it fire); on non-SSM models the old and new paths must behave
    identically (the has_view=false branch is the old logic).

Maintenance (future pin bumps / fork movement):
  - The fusion matcher is one of the most upstream-touched regions of
    ggml-cuda.cu; re-derive from the tracked fork commit in
    external-sources.toml and run `python -m bigcherry sources check`
    before every pin bump.
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

PROVENANCE = {
    "source-id": "stew675-rdna-boosts",
    "plan-item": "RD13",
    "fork-commit": "0153d580dbc2caa8f29b55ad8ddc7088b4c457dd",
    "fork-commit-title": "CUDA: fuse mul_mat + add through a view node",
    "original-commit": "36270950deb0ba979b131fd49fed721ed1256aec",
    "snapshot-head": "9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22",
    "snapshot-base": "4df29be4f4c3673f428170fda944a5b19f743bb8",
    "adaptations": [],
}


_OLD = """    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;"""

_NEW = """    // mul_mat + add, with an optional view (reshape) node between the matmul and the add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        // view (reshape) between the matmul and the add
        const bool has_view = i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_RESHAPE;

        if (has_view) {
            // use ggml_can_fuse_subgraph: views in the subgraph are allowed here
            const ggml_op ops[3] = { op, GGML_OP_RESHAPE, bias_op };
            const int out_nodes[] = { i + 2 };
            if (!ggml_can_fuse_subgraph(cgraph, i, 3, ops, out_nodes, 1) || cgraph->nodes[i + 1]->src[0] != cgraph->nodes[i]) {
                continue;
            }
        } else {
            if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
                continue;
            }
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[has_view ? i + 2 : i + 1];

        // the add reads the matmul output directly, or through the view
        ggml_tensor * mm_or_view = has_view ? cgraph->nodes[i + 1] : mm_node;

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_or_view) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_or_view) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_or_view) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            // bigcherry: HI82 activation-evidence instrumentation, not part
            // of the ported fork change. See 1205_rd12's identical marker
            // for the rationale (silent host-side fusion selection needs an
            // explicit hit signal for unattended validation to trust).
            //
            // Gated on has_view: this branch (mul_mat + add, no reshape) is
            // the pre-existing upstream fusion path, unchanged by this
            // patch -- firing the marker here regardless would falsely
            // report RD13's actual new functionality (the RESHAPE-mediated
            // fusion) as "executed" on ANY ordinary transformer, defeating
            // the whole point of activation evidence. Found via GPT review,
            // req_cc5af49494fe457a.
            if (has_view && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
                static std::atomic_flag bigcherry_rd13_logged = ATOMIC_FLAG_INIT;
                if (!bigcherry_rd13_logged.test_and_set(std::memory_order_relaxed)) {
                    GGML_LOG_INFO("BIGCHERRY_PATCH_HIT patch=1206_rd13 path=mul_mat_add_view_fusion_f\\n");
                }
            }
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = has_view ? 3 : 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            if (has_view && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
                static std::atomic_flag bigcherry_rd13_logged_q = ATOMIC_FLAG_INIT;
                if (!bigcherry_rd13_logged_q.test_and_set(std::memory_order_relaxed)) {
                    GGML_LOG_INFO("BIGCHERRY_PATCH_HIT patch=1206_rd13 path=mul_mat_add_view_fusion_q\\n");
                }
            }
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = has_view ? 3 : 2;"""


PATCH = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="Accept one RESHAPE view between mul_mat and add in the "
                "ggml_cuda_try_fuse matmul+bias fusion "
                "(rdna-boosts 0153d580d / RD13)",
    edits=(
        Edit(
            id="rd13-mul_mat_add_view",
            # The block's first line is a comment; strip_noise blanks comments
            # to spaces (offsets preserved), so it matches a space run. The
            # replacement text restores the fork's updated comment.
            anchor="    " + r"[ ]{12,24}" + re.escape(_OLD[20:]),
            # 52-line block replacement; the default 40-line span guard is
            # a runaway-prevention, not a size limit -- raise it explicitly.
            max_span_lines=60,
            rationale="ggml_cuda_try_fuse: the mul_mat + add / "
                      "mul_mat_id + add_id fusion branch, which today "
                      "only matches the add directly after the matmul",
            mode="replace",
            text=_NEW,
            # Guard on the fork's unique post-change comment line (the
            # non-view branches are byte-identical to the pre-patch ones).
            guard=r"mul_mat \+ add, with an optional view \(reshape\) node between the matmul and the add",
        ),
    ),
)

PATCHES = [PATCH]
