"""Radiance 1.7.0: optionally draft MTP rounds through the trunk lm_head."""

from bigcherry.patcher import Edit, FilePatch

PATCH = FilePatch(
    path="arch/common/rad_block_mtp_fp8.h",
    description="Add RAD_MTP_FULL_HEAD to bypass the quantized two-bit draft-head name map.",
    edits=(
        Edit(
            id="include-cstdlib",
            anchor=r'^#include "rad_block_vit\.h"$',
            text="""
#include <cstdlib>""",
            guard=r"^#include <cstdlib>$",
            expect_matches=1,
            rationale="Radiance 1.4.0 and later drop this header's own <cstdlib>; the switch uses std::getenv/std::atoi, so the patch includes it instead of relying on a transitive include.",
        ),
        Edit(
            id="mtp-full-head-switch",
            anchor=r"^[ \t]*if \(!head \|\| !lm\) return false;$",
            text="""
    /* BigCherry: RAD_MTP_FULL_HEAD=1 drafts through the trunk head (int8 here) instead of the
     * 2-bit copy, trading draft-round traffic for acceptance. */
    if (const char* e = std::getenv("RAD_MTP_FULL_HEAD"); e && std::atoi(e) != 0) return false;""",
            guard=r"RAD_MTP_FULL_HEAD",
            expect_matches=1,
            rationale="The unique null-name guard is the first executable statement in mtp_draft_head_q2, before map_copy creates the draft-head alias.",
        ),
    ),
)
