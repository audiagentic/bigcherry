"""1303: separate tensor-split vector for full-attention tensors + KV cache under -sm tensor.

llama_meta_device_get_split_state splits every tensor with the model's -ts; for Flash-Next that couples the KV
cache (2 heads, rotated over 3 GPUs) to the expert weights, so filling one GPU's spare VRAM with KV is
impossible without also moving weights. BIGCHERRY_ATTN_TS selects a second vector (and BIGCHERRY_ATTN_ROTATE=0
an unrotated assignment) for the full-attention family only; recurrent layers that reuse attn_qkv/attn_gate
names keep -ts because their partition is anchored to ssm_out.weight.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_INCLUDE = "#include <cstring>\n"

_PATTERNS_HEAD = '    static const std::regex pattern_q_weight        ("blk\\\\.\\\\d*\\\\.attn_q.weight");\n'

_CONFIG = r"""    // bigcherry 1303: optional separate split vector for the full-attention family + KV cache.
    struct bigcherry_attn_split_config {
        bool enabled = false;
        bool rotate  = true;
        std::string raw;
        std::vector<float> split;
    };
    static const bigcherry_attn_split_config bigcherry_attn_split = [] {
        bigcherry_attn_split_config cfg;
        const char * env = std::getenv("BIGCHERRY_ATTN_TS");
        if (env == nullptr || env[0] == '\0') {
            return cfg;
        }
        cfg.enabled = true;
        cfg.raw = env;
        if (cfg.raw.front() == ',' || cfg.raw.back() == ',') {
            throw std::runtime_error("BIGCHERRY_ATTN_TS: invalid comma-separated tensor split");
        }
        std::stringstream ss(cfg.raw);
        std::string item;
        while (std::getline(ss, item, ',')) {
            std::stringstream value_stream(item);
            float value = 0.0f;
            if (!(value_stream >> value)) {
                throw std::runtime_error("BIGCHERRY_ATTN_TS: invalid value '" + item + "'");
            }
            value_stream >> std::ws;
            if (!value_stream.eof() || !std::isfinite(value) || value < 0.0f) {
                throw std::runtime_error("BIGCHERRY_ATTN_TS: invalid value '" + item + "'");
            }
            cfg.split.push_back(value);
        }
        if (const char * env_rotate = std::getenv("BIGCHERRY_ATTN_ROTATE")) {
            if (strcmp(env_rotate, "0") == 0) {
                cfg.rotate = false;
            } else if (strcmp(env_rotate, "1") != 0 && env_rotate[0] != '\0') {
                throw std::runtime_error("BIGCHERRY_ATTN_ROTATE must be 0 or 1");
            }
        }
        return cfg;
    }();
    if (bigcherry_attn_split.enabled) {
        if (bigcherry_attn_split.split.size() != ud->n_devices) {
            throw std::runtime_error("BIGCHERRY_ATTN_TS has " + std::to_string(bigcherry_attn_split.split.size()) +
                                     " entries, expected " + std::to_string(ud->n_devices));
        }
        static bool bigcherry_attn_split_logged = false;
        if (!bigcherry_attn_split_logged) {
            bigcherry_attn_split_logged = true;
            LLAMA_LOG_WARN("BIGCHERRY_PATCH_HIT attn_ts=%s attn_rotate=%d\n",
                           bigcherry_attn_split.raw.c_str(), bigcherry_attn_split.rotate ? 1 : 0);
        }
    }

"""

_TC_OLD = "    tensor_config tc = get_tensor_config();\n    split_state.axis = tc.axis;\n"

_TC_NEW = r"""    tensor_config tc = get_tensor_config();
    // bigcherry 1303: full-attention family (never recurrent layers, which reuse attn_qkv/attn_gate names and
    // are anchored to ssm_out) takes BIGCHERRY_ATTN_TS with one shared rotation; mirrored tensors never get here.
    const bool bigcherry_use_attn_split = bigcherry_attn_split.enabled &&
        tc.il < hparams.n_layer_all && !hparams.is_recr(tc.il) && (
            std::regex_match(tensor_name, pattern_q_weight)          ||
            std::regex_match(tensor_name, pattern_kv_weight)         ||
            std::regex_match(tensor_name, pattern_qkv_weight)        ||
            std::regex_match(tensor_name, pattern_q_bias)            ||
            std::regex_match(tensor_name, pattern_kv_bias)           ||
            std::regex_match(tensor_name, pattern_qkv_bias)          ||
            std::regex_match(tensor_name, pattern_qk_norm)           ||
            std::regex_match(tensor_name, pattern_kv_cache)          ||
            std::regex_match(tensor_name, pattern_attn_sinks)        ||
            std::regex_match(tensor_name, pattern_attn_out_weight)   ||
            std::regex_match(tensor_name, pattern_attn_out_bias)     ||
            std::regex_match(tensor_name, pattern_attn_out_a_weight) ||
            std::regex_match(tensor_name, pattern_attn_out_b_weight) ||
            std::regex_match(tensor_name, pattern_attn_q_b_weight)   ||
            std::regex_match(tensor_name, pattern_attn_gate_weight));
    const size_t split_rotation = bigcherry_use_attn_split && !bigcherry_attn_split.rotate ? 0 : tc.rotation;
    split_state.axis = tc.axis;
"""

_TS_OLD = "        const float * tensor_split = ud->model->tensor_split();\n"
_TS_NEW = ("        const float * tensor_split = bigcherry_use_attn_split\n"
           "            ? bigcherry_attn_split.split.data() : ud->model->tensor_split();\n")

PATCHES = [
    FilePatch(
        path="src/llama-model.cpp",
        description="1303: BIGCHERRY_ATTN_TS / BIGCHERRY_ATTN_ROTATE split for attention + KV under -sm tensor",
        language="none",
        edits=(
            Edit(
                id="attn-ts-include",
                anchor=re.escape(_INCLUDE),
                mode="insert_after",
                text="#include <cstdlib>  // bigcherry 1303: std::getenv\n",
                guard=r"#include <cstdlib>  // bigcherry 1303",
                rationale="std::getenv for the env-selected attention split.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="attn-ts-config",
                anchor=re.escape(_PATTERNS_HEAD),
                mode="insert_before",
                text=_CONFIG,
                guard=r"struct bigcherry_attn_split_config \{",
                rationale="Parse the attention split once at the head of llama_meta_device_get_split_state's pattern table.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="attn-ts-select",
                anchor=re.escape(_TC_OLD),
                mode="replace",
                text=_TC_NEW,
                guard=r"const bool bigcherry_use_attn_split =",
                rationale="Classify the tensor once its config (layer index) is known; pick the vector and rotation.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="attn-ts-vector",
                anchor=re.escape(_TS_OLD),
                mode="replace",
                text=_TS_NEW,
                guard=r"\? bigcherry_attn_split\.split\.data\(\) : ud->model->tensor_split\(\);",
                rationale="The split vector feeding the cumulative scan.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="attn-ts-rotation",
                anchor=re.escape("(j + tc.rotation) % ud->n_devices"),
                mode="replace_all",
                text="(j + split_rotation) % ud->n_devices",
                guard=r"\(j \+ split_rotation\) % ud->n_devices",
                rationale="All three rotation uses in the scan/assignment take the family's (possibly unrotated) value.",
                expect_matches=3,
                max_span_lines=1,
            ),
        ),
    ),
]
