"""1306: tensor-split the Qwen4Exp shared expert instead of mirroring it (BIGCHERRY_SHEXP_SPLIT=1).

Upstream llama_meta_device_get_split_state gives ffn_{up,gate,down}_shexp split axes only for DSV4; for Qwen4Exp
they fall through to MIRRORED, so under -sm tensor every GPU holds and computes the whole shared expert every
layer (GPT review req_07434b19f4534cb3). This applies the DSV4 layout to Qwen4Exp: up/gate split on axis 1 and down
on axis 0, all anchored to ffn_down_shexp.weight. The shared-expert output becomes a partial sum that joins the
routed experts' partial output ahead of the same AllReduce, so no extra collective is added; each GPU computes and
stores ~1/n of the shared expert. ffn_gate_inp_shexp (the scalar gate) stays mirrored. Upstream's FFN granularity
rule already covers the shexp patterns. Env-gated for A/B (one binary, both arms).
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_OUTPUT_HEAD = "        // output\n        if (std::regex_match(tensor_name, pattern_output_weight)) {\n"

_SHEXP = """\
        // bigcherry 1306: split the Qwen4Exp shared expert like DSV4 instead of mirroring it on every GPU.
        if (ud->model->arch == LLM_ARCH_QWEN4EXP) {
            static const bool bigcherry_shexp_split = [] {
                const char * s = std::getenv("BIGCHERRY_SHEXP_SPLIT");
                return s != nullptr && std::atoi(s) != 0;
            }();
            if (bigcherry_shexp_split) {
                if (std::regex_match(tensor_name, pattern_ffn_up_shexp_weight) ||
                        std::regex_match(tensor_name, pattern_ffn_gate_shexp_weight)) {
                    return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down_shexp.weight");
                }
                if (std::regex_match(tensor_name, pattern_ffn_down_shexp_weight)) {
                    static std::atomic<bool> bigcherry_shexp_logged{false};
                    if (!bigcherry_shexp_logged.exchange(true)) {
                        LLAMA_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1306_shexp_split\\n");
                    }
                    return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down_shexp.weight");
                }
            }
        }

"""

PATCHES = [
    FilePatch(
        path="src/llama-model.cpp",
        description="1306: Qwen4Exp shared expert split (DSV4 layout) instead of mirrored, env-gated",
        language="none",
        edits=(
            Edit(
                id="qwen4exp-shexp-split",
                anchor=re.escape(_OUTPUT_HEAD),
                mode="insert_before",
                text=_SHEXP,
                guard=r"bigcherry 1306: split the Qwen4Exp shared expert",
                rationale="After the generic FFN rules and before the output-head rules in get_tensor_config; shexp "
                          "tensors otherwise reach the final MIRRORED fallthrough.",
                expect_matches=1,
                max_span_lines=3,
            ),
        ),
    ),
]
