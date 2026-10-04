"""1308: Qwen4Exp recurrent rollback snapshots - copy the strided tail straight into the cache (QFP13).

build the GDN conv state ([TAG_RECURRENT_ROLLBACK_SPLITS] in qwen4exp.cpp) writes one snapshot per rollback slot
(n_rs_seq + 1 = 4 under MTP3) per recurrent layer as ggml_cpy(ggml_cont(tail), dst): the CONT of the strided
tail view is one device copy and the CPY into the cache another, so 8 copy kernels per layer. The kernel census on
Flash-Next decode attributes ~89 of the ~108 __amd_rocclr_copy* launches per generated token to exactly this
sequence (concat_non_cont -> 8 copies -> get_rows). ggml_cpy accepts a non-contiguous source, so with
BIGCHERRY_ROLLBACK_NO_CONT=1 the tail is copied directly: one kernel per slot instead of two. The copy is exact, so
outputs must be bit-identical; the open question this gates is whether every backend/split mode accepts the
strided source (the meta backend's CPY handling), hence env-gated for A/B.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_OLD = "        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst));\n"
_NEW = """\
        // bigcherry 1308: BIGCHERRY_ROLLBACK_NO_CONT=1 copies the strided tail straight into the cache (one kernel
        // per slot instead of CONT + CPY).
        static const bool bigcherry_rollback_no_cont = [] {
            const char * s = std::getenv("BIGCHERRY_ROLLBACK_NO_CONT");
            return s != nullptr && std::atoi(s) != 0;
        }();
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, bigcherry_rollback_no_cont ? tail : ggml_cont(ctx0, tail), dst));
"""

PATCHES = [
    FilePatch(
        path="src/models/qwen4exp.cpp",
        description="1308: rollback snapshot copies without the intermediate CONT (env-gated)",
        language="none",
        edits=(
            Edit(
                id="rollback-no-cont-include",
                anchor=re.escape("#include <algorithm>\n"),
                mode="insert_before",
                text="#include <cstdlib>  // bigcherry 1308: std::getenv / std::atoi\n",
                guard=r"#include <cstdlib>  // bigcherry 1308",
                rationale="First standard-library include of qwen4exp.cpp.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="rollback-no-cont",
                anchor=re.escape(_OLD),
                mode="replace",
                text=_NEW,
                guard=r"bigcherry 1308: BIGCHERRY_ROLLBACK_NO_CONT",
                rationale="The per-slot snapshot write in the recurrent conv-state builder "
                          "([TAG_RECURRENT_ROLLBACK_SPLITS]); the only cpy(cont(tail), dst) in the file.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_ROLLBACK_NO_CONT', '0|1', '0',
           'Qwen4Exp speculative rollback copies state without an extra ggml_cont'),
)
