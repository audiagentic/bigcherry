"""VTP57: the uneven ownership path accepts a vocabulary-weighted head."""
from __future__ import annotations
import re
from bigcherry.patcher import Edit, FilePatch

PATCHES = (
    FilePatch(path='arch/qwen35_fp8/qwen35_fp8.cpp', language="none",
              description='ownership path: vocabulary-weighted head', edits=(
        Edit(id='owned-vhead', anchor=re.escape('        if (m.n_vcls || !m.w_lm_head) {\n            fprintf(stderr, "radiance: qwen35: --tp-ownership reads a plain output.weight, not a "\n                            "vocabulary-weighted head\\n");\n            return RAD_E_UNSUPPORTED;\n        }\n        RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_tok, g.vocab_off, g.vocab_off + g.n_vocab));\n        RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_lm_head, g.vocab_off, g.vocab_off + g.n_vocab));\n'), mode="replace", text='        /* BC_VTP57_OWNED_VHEAD: a vocabulary-weighted head\'s classes hold equal rows on every rank\n         * (rows a multiple of 256 x world), so they shard like the even split; the token table and\n         * the id-to-row map are vocabulary-sized and take the rank\'s share explicitly. */\n        if (!m.n_vcls && !m.w_lm_head) {\n            fprintf(stderr, "radiance: qwen35: --tp-ownership found neither a plain output.weight "\n                            "nor a vocabulary-weighted head\\n");\n            return RAD_E_UNSUPPORTED;\n        }\n        RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_tok, g.vocab_off, g.vocab_off + g.n_vocab));\n        if (m.n_vcls)\n            RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_vrow, g.vocab_off, g.vocab_off + g.n_vocab));\n        else\n            RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_lm_head, g.vocab_off, g.vocab_off + g.n_vocab));\n',
             guard=re.escape('BC_VTP57_OWNED_VHEAD'), expect_matches=1, max_span_lines=10,
             rationale='long-anchor: the refusal block of the ownership vocabulary declare is replaced as one unit'),
    )),
)
