#ifndef RAD_BLOCK_MOE_FP8_H
#define RAD_BLOCK_MOE_FP8_H

#include "rad_fp8.h"
#include "rad_device.h"

struct MoeFP8 {
    std::vector<rad_weight> w_gu, w_gus, w_dn, w_dns;
    rad_op op_sgate = 0;
    rad_op op_ar = 0, op_add = 0;

    int declare() {
        w_gu.resize((size_t)ne); w_gus.resize((size_t)ne);
        w_dn.resize((size_t)ne); w_dns.resize((size_t)ne);

        w_gu[(size_t)e] = decl_w();
        if (!w_gu[(size_t)e]) return RAD_E_INVAL;
        w_dn[(size_t)e] = decl_w();
        if (!w_dn[(size_t)e]) return RAD_E_INVAL;

        rad_weight& h = w_gu[(size_t)s];
        h = decl_view();
        if (!h) return RAD_E_INVAL;
        RAD_ARCH_TRY(span(h, q, 1));

        h = w_gus[(size_t)s];
        if (!h) return RAD_E_INVAL;
        RAD_ARCH_TRY(span(h, q, sdiv));

        h = w_dn[(size_t)s];
        if (!h) return RAD_E_INVAL;
        RAD_ARCH_TRY(span(h, q, 1));

        h = w_dns[(size_t)s];
        if (!h) return RAD_E_INVAL;
        RAD_ARCH_TRY(span(h, q, sdiv));

        w_pgu[(size_t)j] = decl_w();
        if (!w_pgu[(size_t)j]) return RAD_E_INVAL;
        w_pdn[(size_t)j] = decl_w();
        if (!w_pdn[(size_t)j]) return RAD_E_INVAL;

        op_add = rw(b, op, {w.x, w.h.x}, {w.x});
        return RAD_OK;
    }

    void pass(RadCtx* c_, int64_t T, int64_t r0, int64_t rows) const {
        const int64_t ne = c.n_expert, ne_all = c.n_expert_all, k = c.top_k;
        const bool shared = op_sgate != 0;
        const bool rfused = op_router_fused && rows <= router_fused_rows;
        if (rfused) {
        } else if (op_topk_scatter && rows <= topk_scatter_rows) {
            use_fused();
        } else {
            RAD_ISSUE_N(c_, op_topk, rows, brow_slice(w.logits, r0, rows, ne_all),
                        brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k));
            RAD_ISSUE_N(c_, op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                        brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
        }

        if (!gather_taken(g, T, gather_out_rows, gather_out_rows6))
            RAD_ISSUE_N(c_, op_gather, rows, brows(w.edn, rows * k), brow_slice(w.ew, r0, rows, k),
                        brows(w.sorted, rows * k),
                        gfold ? brow_slice(w.sout, r0, rows, g.n_embd) : RAD_NONE,
                        gfold ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE,
                        brow_slice(w.h.x, r0, rows, g.n_embd));

        if (op_ar) issue_ar();
    }
};
