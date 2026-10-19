"""Radiance 1.7.0: remote-MoE shadow compare and mask-only diagnostics, for a diagnostic build only.

Requires rad0002 (the extension point). Every edit is a call site, compiled only under BC_MOE_REMOTE_DIAG, into the
diagnostic header this package creates; without the macro the file is not even included. No production source,
patch set or experiment may include this package (patch-lint, gates.diagnostic_build_problems).
"""

from bigcherry.patcher import Edit, FilePatch

_BC_MOE_REMOTE_DIAG_H = r"""/* bc_moe_remote_diag.h -- remote-MoE diagnostics for the MoE block (patch rad0004). Diagnostic builds only.
 *
 * shadow:    the helper computes the owned routes and compares them with the local result, while the local result
 *            still goes forward (R3_MOE_WORKER_SHADOW); the helper reports the per-layer error at shutdown.
 * mask-only: the local result is captured, the routes are published and masked late, the local rows are restored
 *            to engine order, and the helper's rows are added to the captured ones (R3_MOE_WORKER_MASK_ONLY). */
#ifndef BC_MOE_REMOTE_DIAG_H
#define BC_MOE_REMOTE_DIAG_H

#ifndef BC_MOE_REMOTE_DIAG
#error "bc_moe_remote_diag.h belongs to the diagnostic build: define BC_MOE_REMOTE_DIAG (patch rad0004)"
#endif

#include "bc_moe_remote_block.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace rad {
namespace arch {

struct BcMoeRemoteDiag {
    rad_op shadow = 0, mask_capture = 0, mask_restore = 0;

    /* After BcMoeRemote::declare: declare the three diagnostic ops when the step gate admits any step. */
    template <class Moe>
    int declare(RadBuilder* b, const Moe& m, const std::vector<int64_t>& reg, int layer) {
        const auto& c = m.c;
        const auto& g = m.g;
        const auto& w = m.w;
        if (!bc_moe_remote_wanted_upto(g.max_tok)) return RAD_OK;
        if (!bc_moe_remote_diag_build()) {
            std::fprintf(stderr, "radiance: the MoE block was built with BC_MOE_REMOTE_DIAG but the engine core was "
                                 "not; build both with the diagnostic flag\n");
            return RAD_E_INVAL;
        }
        shadow = rw(b, RAD_OP(b, "moe_mailbox_shadow",
                              RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd), RAD_INT("top_k", c.top_k)),
                              RAD_NOWEIGHTS),
                    {w.sorted, w.edn, w.ew}, {w.edn});
        mask_capture = rw(b, RAD_OP(b, "moe_mailbox_mask_ref",
                                    RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                               RAD_INT("top_k", c.top_k), RAD_INT("restore", 0)),
                                    RAD_NOWEIGHTS),
                          {w.sorted, w.edn}, {w.edn});
        mask_restore = rw(b, RAD_OP(b, "moe_mailbox_mask_ref",
                                    RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                               RAD_INT("top_k", c.top_k), RAD_INT("restore", 1)),
                                    RAD_NOWEIGHTS),
                          {w.sorted, w.edn}, {w.edn});
        if (!shadow || !mask_capture || !mask_restore) {
            std::fprintf(stderr, "radiance: MoE remote diagnostic ops not declared at layer %d (shadow %d mask %d/%d)\n",
                         layer, shadow != 0, mask_capture != 0, mask_restore != 0);
            return RAD_E_INVAL;
        }
        if (bc_moe_remote_shadow() && reg.size() < (size_t)c.n_expert) {
            std::string pl;
            for (int64_t e = 0; e < c.n_expert; ++e)
                if (std::find(reg.begin(), reg.end(), e) == reg.end())
                    pl += (pl.empty() ? "" : ",") + std::to_string(e);
            std::fprintf(stderr, "radiance: MoE remote shadow: layer %d protected [%s] n_reg %lld ncls %d\n", layer,
                         pl.c_str(), (long long)reg.size(), m.ncls);
        }
        return RAD_OK;
    }

    /* Before the shared arm's down projection, once every routed and protected down has written `edn`. */
    template <class Moe>
    void after_down(RadCtx* c_, const Moe& m, const BcMoeRemote& base, int64_t T, int64_t r0, int64_t rows,
                    const RadOperand& ecnt) const {
        const bool mask_only = base.active(T) && bc_moe_remote_mask_only();
        const bool compare = base.policy(T) && bc_moe_remote_shadow();
        const auto& c = m.c;
        const auto& g = m.g;
        const auto& w = m.w;
        const int64_t k = c.top_k;
        if (mask_only) {
            RAD_ISSUE_N(c_, mask_capture, rows, brows(w.sorted, rows * k), brows(w.edn, rows * k));
            const rad_buf q = c.expert_rot ? w.hr.q : w.h.q;
            const rad_buf s = c.expert_rot ? w.hr.s : w.h.s;
            RAD_ISSUE_N(c_, base.publish, rows,
                        brow_slice(w.ids, r0, rows, k),
                        brow_slice(w.ew, r0, rows, k),
                        brow_slice(w.h.x, r0, rows, g.n_embd),
                        brow_slice(q, r0, rows, g.n_embd),
                        brow_slice(s, r0, rows, g.n_embd / RAD_FP8_BLOCK),
                        brows(q, T));
            RAD_ISSUE_N(c_, m.op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                        brows(w.sorted, rows * k), brows(w.eoff, c.n_expert + 1), ecnt);
            RAD_ISSUE_N(c_, mask_restore, rows, brows(w.sorted, rows * k), brows(w.edn, rows * k));
        }
        if (compare)
            RAD_ISSUE_N(c_, shadow, rows, brows(w.sorted, rows * k), brows(w.edn, rows * k),
                        brow_slice(w.ew, r0, rows, k));
    }
};

}  /* namespace arch */
}  /* namespace rad */

#endif /* BC_MOE_REMOTE_DIAG_H */
"""

PATCHES = (
    FilePatch(
        path="arch/common/bc_moe_remote_diag.h",
        create=True,
        description="Create the remote-MoE diagnostics (shadow compare, mask-only) behind the extension point.",
        edits=(Edit(
            id="create-bc-moe-remote-diag",
            anchor=r"\A",
            text=_BC_MOE_REMOTE_DIAG_H,
            guard=r"^struct BcMoeRemoteDiag \{$",
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="arch/common/rad_block_moe_fp8.h",
        description="Call the diagnostics from the MoE block, compiled only under BC_MOE_REMOTE_DIAG.",
        edits=(
            Edit(
                id="diag-include",
                anchor=r'^#include "bc_moe_remote_block\.h"$',
                text='\n#ifdef BC_MOE_REMOTE_DIAG\n#include "bc_moe_remote_diag.h"\n#endif',
                guard=r'^#include "bc_moe_remote_diag\.h"$',
                expect_matches=1,
                rationale="rad0002's extension-point include is the unique anchor; the diagnostic header rides beside it.",
            ),
            Edit(
                id="diag-member",
                anchor=r"^[ \t]*BcMoeRemote bc_remote;$",
                text="\n#ifdef BC_MOE_REMOTE_DIAG\n    BcMoeRemoteDiag bc_diag;\n#endif",
                guard=r"^[ \t]*BcMoeRemoteDiag bc_diag;$",
                expect_matches=1,
                rationale="The diagnostic state sits beside the extension point's own member.",
            ),
            Edit(
                id="diag-declare",
                anchor=r"^[ \t]*RAD_ARCH_TRY\(bc_remote\.declare\(b, \*this, reg, l\)\);$",
                text="\n#ifdef BC_MOE_REMOTE_DIAG\n    RAD_ARCH_TRY(bc_diag.declare(b, *this, reg, l));\n#endif",
                guard=r"RAD_ARCH_TRY\(bc_diag\.declare\(b, \*this, reg, l\)\);",
                expect_matches=1,
                rationale="The diagnostic ops are declared right after the mailbox ops they accompany.",
            ),
            Edit(
                id="diag-after-down",
                anchor=r"^[ \t]*if \(shared\) sh_down\.step\(c_, w\.sff, w\.sout, T, r0, rows\);$",
                mode="insert_before",
                text="#ifdef BC_MOE_REMOTE_DIAG\n    bc_diag.after_down(c_, *this, bc_remote, T, r0, rows, ecnt);\n#endif\n",
                guard=r"bc_diag\.after_down\(c_, \*this, bc_remote, T, r0, rows, ecnt\);",
                expect_matches=1,
                rationale="The shared-down issue is the first code line after every routed and protected down writer, so mask-only can snapshot the engine rows before masking and restore local row order after the diagnostic rescatter.",
            ),
        ),
    ),
)
