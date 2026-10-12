"""Opt-in MTP draft logits from the full trunk lm_head (default off)."""

from bigcherry.patcher import Edit, FilePatch

PATCH = FilePatch(
    path="arch/common/rad_block_mtp_fp8.h",
    description="Select full head before declaring quantized draft head",
    edits=(
        Edit(
            id="full-head-switch",
            anchor=r"^    if \(!head \|\| !lm\) return false;$",
            mode="insert_after",
            text="""
    /* BigCherry: RAD_MTP_FULL_HEAD=1 drafts through the trunk head (int8 here) instead of the
     * 2-bit copy, trading draft-round traffic for acceptance. */
    if (const char* e = std::getenv("RAD_MTP_FULL_HEAD"); e && std::atoi(e) != 0) return false;""",
            guard=r"if \(const char\* e = std::getenv\(\"RAD_MTP_FULL_HEAD\"\); e && std::atoi\(e\) != 0\)",
            expect_matches=1,
            rationale="Unique null-head guard at start of mtp_draft_head_q2",
        ),
    ),
)
