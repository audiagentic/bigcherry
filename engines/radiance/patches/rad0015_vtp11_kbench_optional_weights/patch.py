"""VTP11: keep rad-kbench's fixed-weight cursor aligned across absent optionals.

The engine graph has a required GDN convolution weight, an optional absent
convolution bias, then required A_log and dt_bias. The old benchmark advanced
the declared weight cursor for the absent bias, so it silently skipped all 28
GDN oracle cases despite reporting exit zero. Count present fixed operands
first; only advance the cursor for operands actually drawn. This is a
diagnostic-harness fix, not a serving/placement change.
"""

import re

from bigcherry.patcher import Edit, FilePatch

PATCH = FilePatch(
    path="tools/rad_kbench.cpp",
    language="none",  # Exact pinned-source hunk includes the old comment.
    description="Keep kbench weight binding aligned when an optional fixed-weight slot is absent",
    edits=(
        Edit(
            id="vtp11-kbench-present-weight-contract",
            anchor=re.escape(
                """                std::vector<std::vector<uint8_t>> bufs(sh.size()), ref_out(sh.size());"""
            ),
            mode="insert_before",
            text="""                /* BC_VTP11_KBENCH_PRESENT_WEIGHT_CONTRACT: shape fixtures can omit an
                 * optional fixed weight in the middle (GDN's conv bias), while the
                 * architecture declares only the present weights. Never map A_log
                 * onto the absent bias or silently drop dt_bias. A weight table has
                 * a separate multi-handle contract and is checked separately. */
                int present_fixed_weights = 0;
                bool has_weight_table = false;
                for (size_t wpos = 0; wpos < sh.size(); ++wpos) {
                    const int wrole = o.schema->operands[wpos].role;
                    if (wrole == RAD_OPD_WTAB) has_weight_table = true;
                    if (wrole == RAD_OPD_WEIGHT && !sh[wpos].absent)
                        ++present_fixed_weights;
                }
                if (!has_weight_table && present_fixed_weights != (int)o.weights.size()) {
                    skipped_ops[o.op + "  (fixture has " +
                                std::to_string(present_fixed_weights) +
                                " present fixed weights but declare bound " +
                                std::to_string(o.weights.size()) +
                                "; cannot map optional weights unambiguously)"]++;
                    ++st.skipped;
                    continue;
                }
""",
            guard=r"BC_VTP11_KBENCH_PRESENT_WEIGHT_CONTRACT",
            expect_matches=1,
            rationale="One benchmark-case allocation point, after shape draw and byte-budget check",
        ),
        Edit(
            id="vtp11-kbench-absent-weight-cursor",
            anchor=re.escape(
                """                        /* An absent weight still consumes its handle (the list is positional). */
                        if (role == RAD_OPD_WEIGHT) ++wk;"""
            ),
            mode="replace",
            text="""                        /* BC_VTP11_KBENCH_ABSENT_WEIGHT_CURSOR: an absent optional
                         * fixed-weight slot has no declared handle. The present-
                         * weight count above refuses ambiguous binding instead of
                         * shifting every later handle (GDN A_log/dt_bias). */""",
            guard=r"BC_VTP11_KBENCH_ABSENT_WEIGHT_CURSOR",
            expect_matches=1,
            rationale="Unique optional-weight handling in benchmark draw, not engine runtime",
        ),
    ),
)
