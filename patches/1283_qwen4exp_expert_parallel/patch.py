"""1283 (MET04): whole-expert parallelism inside the tensor split, one AllReduce per MoE block.

With -sm tensor every weight is sliced across the devices. For the routed experts that slice is along the rows, so
each device computes a part of EVERY selected expert. With BIGCHERRY_MOE_EP=1 the expert weights ffn_gate_exps /
ffn_up_exps / ffn_down_exps are sliced along the expert index instead: a device holds whole experts
[base_d, base_d + n_d) (sizes from -ts, granularity one expert) and runs the range MUL_MAT_ID of patch 1281 - it
computes the selected experts it holds and writes exact zeros for the others.

Three pieces:
  1. src/llama-model.cpp: the split state of the three expert weights is AXIS_2 with granularity 1 under the flag.
  2. ggml-backend-meta.cpp, split states: MUL_MAT_ID with src0 on AXIS_2 and a replicated activation is PARTIAL (the
     per-device results sum to the full result). Each device's node gets 1281's range marker and its id base.
  3. ggml-backend-meta.cpp, subgraph cutting: the AllReduce after the first expert projection is DELAYED through the
     block: gate, up -> GLU -> down, and from there the existing delay (weighting MULs, the per-expert VIEW + ADD sum)
     carries it to the block output. One AllReduce per block, as with the row split.

Why the delay is valid: on device d a lane whose expert is outside d's range is an exact +0 after gate and after up
(range op), GLU(0, 0) = 0, and the down range op writes +0 for that lane whatever its input. A lane whose expert is on
d is computed entirely on d. Every expert is on exactly one device, so the per-device block outputs sum to the exact
result; everything after down is linear (review req_5ce0760adbb8433f).

The delay is a strict pattern match and falls back to an immediate AllReduce (three per block) when the block does
not have the expected shape: the two projections must be consecutive range nodes over the same activation and ids,
feed one GLU and nothing else, and the down projection must take that GLU and the same ids with the same per-device
expert ranges. Expert biases (ADD_ID has no range form), the merged gate_up tensor and scale MULs are not matched.

Requires 1281 with its GPU paths (phase B). Off by default; nothing changes without the flag.
"""
import re as _re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "core"
STATE = "untested"

# ---- src/llama-model.cpp -------------------------------------------------------------------------------------------
# getenv / atoi need <cstdlib>, which this file does not include at b11402 (review req_f0dac9a3c42f4d88)
_A_MODEL_INC = "#include <cstdint>\n"
_N_MODEL_INC = _A_MODEL_INC + "#include <cstdlib>  // BigCherry 1283: getenv\n"

_A_MODEL_PAT = '    static const std::regex pattern_ffn_down_shexp_weight ("blk\\\\.\\\\d*\\\\.ffn_down_shexp.weight");\n'
_N_MODEL_PAT = _A_MODEL_PAT + (
    "    // BigCherry 1283: whole experts per device (split along the expert index) instead of rows of every expert\n"
    '    static const std::regex bc_pattern_ffn_exps_weight("blk\\\\.\\\\d*\\\\.ffn_(gate|up|down)_exps\\\\.weight");\n'
    '    static const bool bc_moe_ep = getenv("BIGCHERRY_MOE_EP") != nullptr && atoi(getenv("BIGCHERRY_MOE_EP")) != 0;\n'
)

_A_MODEL_AXIS = (
    "        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_gate_weight)) {\n"
    '            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");\n'
)
_N_MODEL_AXIS = (
    "        if (bc_moe_ep && std::regex_match(tensor_name, bc_pattern_ffn_exps_weight)) {\n"
    "            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_2); // BigCherry 1283\n"
    "        }\n"
) + _A_MODEL_AXIS

_A_MODEL_GRAN = (
    "        // FFN\n"
    "        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_up_bias) ||\n"
)
_N_MODEL_GRAN = (
    "        if (bc_moe_ep && std::regex_match(tensor_name, bc_pattern_ffn_exps_weight)) {\n"
    "            // BigCherry 1283: experts are indivisible; -ts then distributes the expert count\n"
    "            return std::vector<int64_t>(segments.size(), 1);\n"
    "        }\n"
    "\n"
) + _A_MODEL_GRAN

# ---- ggml-backend-meta.cpp -----------------------------------------------------------------------------------------
_A_META_RULE = (
    "    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {\n"
)
_N_META_RULE = _A_META_RULE + (
    "        // BigCherry 1283: experts split along the expert index. Each device computes the selected experts it holds\n"
    "        // and zeros for the others (range MUL_MAT_ID, 1281), so the per-device results sum to the full result.\n"
    "        if (tensor->op == GGML_OP_MUL_MAT_ID && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2 &&\n"
    "                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {\n"
    "            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};\n"
    "        }\n"
)

_A_META_NODE = "        simple_tensors.push_back(t_ij);\n"
_N_META_NODE = (
    "        if (tensor->op == GGML_OP_MUL_MAT_ID && tensor->src[0] != nullptr && ggml_backend_buffer_is_meta(tensor->src[0]->buffer)) {\n"
    "            // BigCherry 1283: experts split along the expert index - this device holds the experts\n"
    "            // [id_base, id_base + its slice) and its node becomes 1281's range MUL_MAT_ID (ids stay global)\n"
    "            const ggml_backend_meta_split_state bc_ss0 = ggml_backend_meta_get_split_state(tensor->src[0], /*assume_sync =*/ true);\n"
    "            if (bc_ss0.axis == GGML_BACKEND_SPLIT_AXIS_2) {\n"
    "                GGML_ASSERT(bc_ss0.n_segments == 1 && bc_ss0.nr[0] == 1);\n"
    "                int64_t bc_id_base = 0;\n"
    "                for (size_t k = 0; k < j; k++) {\n"
    "                    bc_id_base += bc_ss0.ne[k];\n"
    "                }\n"
    "                GGML_ASSERT(bc_id_base <= INT32_MAX);\n"
    "                ggml_set_op_params_i32(t_ij, 6, 0x52414E47); // GGML_BC_MUL_MAT_ID_RANGE_MARK\n"
    "                ggml_set_op_params_i32(t_ij, 7, (int32_t) bc_id_base);\n"
    "            }\n"
    "        }\n"
    "\n"
) + _A_META_NODE

_A_META_DELAY = (
    "                ggml_tensor * node = cgraph->nodes[id];\n"
    "                int32_t n_used = ggml_node_get_use_count(cgraph, id);\n"
    "\n"
    "                // Skip MIRRORED nodes that don't consume node\n"
)
_N_META_DELAY = (
    "                ggml_tensor * node = cgraph->nodes[id];\n"
    "                int32_t n_used = ggml_node_get_use_count(cgraph, id);\n"
    "\n"
    "                // BigCherry 1283: whole-expert MoE block. `node` is the first expert projection (a MUL_MAT_ID whose\n"
    "                // weights are split along the expert index). Its inactive lanes are exact zeros on every device, they\n"
    "                // stay zero through the second projection, the GLU and the down projection, and each active lane is\n"
    "                // computed wholly on the device that holds its expert - so the AllReduce can wait until after the\n"
    "                // down projection (and from there the existing delay below carries it to the block output).\n"
    "                // Strict pattern, anything else keeps the AllReduce right here.\n"
    "                {\n"
    "                    auto bc_ep_ss = [&](const ggml_tensor * t, ggml_backend_meta_split_state & ss) -> bool {\n"
    "                        if (t->op != GGML_OP_MUL_MAT_ID || t->src[0] == nullptr || !ggml_backend_buffer_is_meta(t->src[0]->buffer)) {\n"
    "                            return false;\n"
    "                        }\n"
    "                        ss = ggml_backend_meta_get_split_state(t->src[0], false);\n"
    "                        return ss.axis == GGML_BACKEND_SPLIT_AXIS_2 && ss.n_segments == 1;\n"
    "                    };\n"
    "                    auto bc_same_ranges = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {\n"
    "                        for (size_t j = 0; j < n_backends; j++) {\n"
    "                            if (a.ne[j] != b.ne[j]) {\n"
    "                                return false;\n"
    "                            }\n"
    "                        }\n"
    "                        return true;\n"
    "                    };\n"
    "                    ggml_backend_meta_split_state bc_ss_a, bc_ss_b, bc_ss_d;\n"
    "                    if (bc_ep_ss(node, bc_ss_a) && id + 3 < cgraph->n_nodes && n_used == 1) {\n"
    "                        ggml_tensor * bc_b = cgraph->nodes[id + 1]; // the other projection\n"
    "                        ggml_tensor * bc_g = cgraph->nodes[id + 2]; // GLU(gate, up)\n"
    "                        ggml_tensor * bc_d = cgraph->nodes[id + 3]; // down projection\n"
    "                        const bool bc_match =\n"
    "                            bc_ep_ss(bc_b, bc_ss_b) && bc_same_ranges(bc_ss_a, bc_ss_b) &&\n"
    "                            bc_b->src[1] == node->src[1] && bc_b->src[2] == node->src[2] &&\n"
    "                            ggml_node_get_use_count(cgraph, id + 1) == 1 &&\n"
    "                            bc_g->op == GGML_OP_GLU &&\n"
    "                            ((bc_g->src[0] == node && bc_g->src[1] == bc_b) || (bc_g->src[0] == bc_b && bc_g->src[1] == node)) &&\n"
    "                            ggml_node_get_use_count(cgraph, id + 2) == 1 &&\n"
    "                            bc_ep_ss(bc_d, bc_ss_d) && bc_same_ranges(bc_ss_a, bc_ss_d) &&\n"
    "                            bc_d->src[1] == bc_g && bc_d->src[2] == node->src[2];\n"
    "                        if (bc_match) {\n"
    "                            node = bc_d;\n"
    "                            id += 3;\n"
    "                            idr = id;\n"
    "                            n_used = ggml_node_get_use_count(cgraph, id);\n"
    "                        }\n"
    "                    }\n"
    "                }\n"
    "\n"
    "                // Skip MIRRORED nodes that don't consume node\n"
)

# The AllReduce scratch is sized from the node where the PARTIAL result first appears. With the delay through an
# expert-index block that node is the first projection ([n_ff, n_used, n_tokens]) while the AllReduce runs on the
# delayed node, which can be larger (the down output is [n_embd, n_used, n_tokens] when the delay stops there).
# From GPT's implementation draft (req_db241e25b4384013).
_A_META_TMP = "                const int i_delayed = get_i_delayed(i);\n"
_N_META_TMP = _A_META_TMP + (
    "                if (i_delayed > i) {\n"
    "                    // BigCherry 1283: the scratch must hold the node the AllReduce actually runs on\n"
    "                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(cgraph->nodes[i_delayed]));\n"
    "                }\n"
)

PATCHES = [
    FilePatch(
        path="src/llama-model.cpp",
        description="1283: BIGCHERRY_MOE_EP=1 splits the routed expert weights along the expert index",
        language="none",
        edits=(
            Edit(id="moe-ep-include", anchor=_re.escape(_A_MODEL_INC), mode="replace", text=_N_MODEL_INC,
                 guard=r"#include <cstdlib>  // BigCherry 1283: getenv", rationale="Standard includes of the file.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="moe-ep-flag", anchor=_re.escape(_A_MODEL_PAT), mode="replace", text=_N_MODEL_PAT,
                 guard=r"static const bool bc_moe_ep = ", rationale="After the FFN tensor-name patterns of the split-state callback.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="moe-ep-axis", anchor=_re.escape(_A_MODEL_AXIS), mode="replace", text=_N_MODEL_AXIS,
                 guard=r"return get_tensor_config_impl\(GGML_BACKEND_SPLIT_AXIS_2\); // BigCherry 1283",
                 rationale="Before the row-split rule for the FFN up / gate weights.", expect_matches=1, max_span_lines=3),
            Edit(id="moe-ep-granularity", anchor=_re.escape(_A_MODEL_GRAN), mode="replace", text=_N_MODEL_GRAN,
                 guard=r"BigCherry 1283: experts are indivisible",
                 rationale="Before the FFN granularity rule (quant block multiples apply to rows, not to experts).",
                 expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-backend-meta.cpp",
        description="1283: expert-index split of MUL_MAT_ID - split state, per-device range nodes, one delayed AllReduce per block",
        language="none",
        edits=(
            Edit(id="moe-ep-split-state", anchor=_re.escape(_A_META_RULE), mode="replace", text=_N_META_RULE,
                 guard=r"tensor->op == GGML_OP_MUL_MAT_ID && src_ss\[0\]\.axis == GGML_BACKEND_SPLIT_AXIS_2",
                 rationale="First rule of handle_mul_mat, ahead of the batched-matmul rule that would return AXIS_2.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="moe-ep-range-node", anchor=_re.escape(_A_META_NODE), mode="replace", text=_N_META_NODE,
                 guard=r"ggml_set_op_params_i32\(t_ij, 6, 0x52414E47\);",
                 rationale="Where a node's per-device tensor is finished, before it is stored.", expect_matches=1,
                 max_span_lines=2),
            Edit(id="moe-ep-delay", anchor=_re.escape(_A_META_DELAY), mode="replace", text=_N_META_DELAY,
                 guard=r"BigCherry 1283: whole-expert MoE block\.",
                 rationale="Head of get_i_delayed_branch, before the existing delay stages.", expect_matches=1,
                 max_span_lines=5),
            Edit(id="moe-ep-tmp-size", anchor=_re.escape(_A_META_TMP), mode="replace", text=_N_META_TMP,
                 guard=r"BigCherry 1283: the scratch must hold the node the AllReduce actually runs on",
                 rationale="Subgraph loop, right after the delayed index is known.", expect_matches=1, max_span_lines=2),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_MOE_EP", "0|1", "0",
           "tensor split: give each device whole routed experts (split along the expert index, sized by -ts) instead "
           "of a row slice of every expert; needs the range MUL_MAT_ID of 1281 on the GPU"),
)
