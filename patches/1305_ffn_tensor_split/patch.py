"""1305: separate tensor-split vector for the FFN/MoE family under -sm tensor (QFP09).

1303 decoupled attention + KV from -ts. The decode census then showed the dominant remaining cost is AllReduce
arrival skew (~80 us median per AR, R9700 last in ~70% of ARs): -ts sizes the R9700's share of the expert weights
by its VRAM, while its memory bandwidth is ~2/3 of an XTX. BIGCHERRY_FFN_TS="a,b,c" gives every ffn_* weight/bias
(routed experts, shared expert, dense FFN) its own split vector so expert placement can follow bandwidth, within
VRAM limits, independently of attention (BIGCHERRY_ATTN_TS) and of everything else (-ts). Each FFN group keeps the
upstream anchors (up/gate follow down, shexp follows down_shexp) and the upstream per-layer rotation, so all members
of a group share one vector and rotation. Router (ffn_gate_inp) and norms are untouched. Requires 1303.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_ROTATION_LINE = (
    "    const size_t split_rotation = bigcherry_use_attn_split && !bigcherry_attn_split.rotate ? 0 : tc.rotation;\n"
)

_FFN_SELECT = r"""    // bigcherry 1305: optional separate split vector for the FFN/MoE family (BIGCHERRY_FFN_TS).
    static const std::vector<float> bigcherry_ffn_split = [] {
        std::vector<float> v;
        const char * env = std::getenv("BIGCHERRY_FFN_TS");
        if (env == nullptr || env[0] == '\0') {
            return v;
        }
        const std::string raw = env;
        if (raw.front() == ',' || raw.back() == ',') {
            throw std::runtime_error("BIGCHERRY_FFN_TS: invalid comma-separated tensor split");
        }
        std::stringstream ss(raw);
        std::string item;
        while (std::getline(ss, item, ',')) {
            std::stringstream value_stream(item);
            float value = 0.0f;
            if (!(value_stream >> value)) {
                throw std::runtime_error("BIGCHERRY_FFN_TS: invalid value '" + item + "'");
            }
            value_stream >> std::ws;
            if (!value_stream.eof() || !std::isfinite(value) || value < 0.0f) {
                throw std::runtime_error("BIGCHERRY_FFN_TS: invalid value '" + item + "'");
            }
            v.push_back(value);
        }
        return v;
    }();
    if (!bigcherry_ffn_split.empty() && bigcherry_ffn_split.size() != ud->n_devices) {
        throw std::runtime_error("BIGCHERRY_FFN_TS has " + std::to_string(bigcherry_ffn_split.size()) +
                                 " entries, expected " + std::to_string(ud->n_devices));
    }
    const bool bigcherry_use_ffn_split = !bigcherry_ffn_split.empty() && !bigcherry_use_attn_split && (
        std::regex_match(tensor_name, pattern_ffn_up_weight)          ||
        std::regex_match(tensor_name, pattern_ffn_up_bias)            ||
        std::regex_match(tensor_name, pattern_ffn_gate_weight)        ||
        std::regex_match(tensor_name, pattern_ffn_gate_bias)          ||
        std::regex_match(tensor_name, pattern_ffn_gate_up_weight)     ||
        std::regex_match(tensor_name, pattern_ffn_down_weight)        ||
        std::regex_match(tensor_name, pattern_ffn_down_bias)          ||
        std::regex_match(tensor_name, pattern_ffn_down_exps_bias)     ||
        std::regex_match(tensor_name, pattern_ffn_up_shexp_weight)    ||
        std::regex_match(tensor_name, pattern_ffn_gate_shexp_weight)  ||
        std::regex_match(tensor_name, pattern_ffn_down_shexp_weight));
    if (bigcherry_use_ffn_split) {
        static std::atomic<bool> bigcherry_ffn_split_logged{false};  // split-state calls can be concurrent
        if (!bigcherry_ffn_split_logged.exchange(true)) {
            std::string bigcherry_ffn_raw;
            for (size_t i = 0; i < bigcherry_ffn_split.size(); i++) {
                bigcherry_ffn_raw += (i ? "," : "") + std::to_string(bigcherry_ffn_split[i]);
            }
            LLAMA_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1305_ffn_ts ffn_ts=%s\n", bigcherry_ffn_raw.c_str());
        }
    }
"""

_VECTOR_OLD = ("        const float * tensor_split = bigcherry_use_attn_split\n"
               "            ? bigcherry_attn_split.split.data() : ud->model->tensor_split();\n")

_VECTOR_NEW = ("        const float * tensor_split = bigcherry_use_attn_split ? bigcherry_attn_split.split.data()\n"
               "            : bigcherry_use_ffn_split ? bigcherry_ffn_split.data() : ud->model->tensor_split();\n")

PATCHES = [
    FilePatch(
        path="src/llama-model.cpp",
        description="1305: BIGCHERRY_FFN_TS split vector for the FFN/MoE family under -sm tensor",
        language="none",
        edits=(
            Edit(
                id="ffn-ts-select",
                anchor=re.escape(_ROTATION_LINE),
                mode="insert_before",
                text=_FFN_SELECT,
                guard=r"bigcherry 1305: optional separate split vector for the FFN/MoE family",
                rationale="1303's rotation line: the attention classification is complete, FFN classification goes next.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="ffn-ts-vector",
                anchor=re.escape(_VECTOR_OLD),
                mode="replace",
                text=_VECTOR_NEW,
                guard=r": bigcherry_use_ffn_split \? bigcherry_ffn_split\.data\(\)",
                rationale="1303's split-vector selection: add the FFN vector as the second choice.",
                expect_matches=1,
                max_span_lines=3,
            ),
        ),
    ),
]
