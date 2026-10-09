"""1278: enable -sm tensor for qwen4exp (Qwen3.8-Flash-Next).

b11233 already carries qwen4exp tensor-split rules (llama-model.cpp: the
qwen3next/qwen35/qwen35moe/qwen4exp Q-gate and SSM/GDN split segments) but
llm_arch_supports_sm_tensor() still rejects the architecture
("TODO: fix test-llama-archs"), so -sm tensor fails at load. This patch removes
qwen4exp from the rejection list. Correctness is validated end to end against
-sm layer (the sparse-attention indexer, per-layer embeddings and hc_ffn
inject paths are not covered by the shared rules).
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "model-arch"
STATE = "untested"

_GATE_LINE = "        case LLM_ARCH_QWEN4EXP:   // TODO: fix test-llama-archs\n"

PATCHES = [
    FilePatch(
        path="src/llama-arch.cpp",
        language="none",
        description="Allow LLAMA_SPLIT_MODE_TENSOR for qwen4exp.",
        edits=(
            Edit(
                id="qwen4exp-sm-tensor-allow",
                anchor=_re.escape(_GATE_LINE),
                text="",
                mode="replace",
                guard=r"case LLM_ARCH_QWEN3TTS:\n            return false;",
                expect_matches=1,
                rationale="qwen4exp is the only entry to drop; the rest of the unsupported list stays.",
            ),
        ),
    ),
]
