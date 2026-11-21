    /* BC_LTX3_C06_ATTN_CORE: use existing issue order. Each rank
     * issues two exact full-width Q/K reductions before its local RoPE. */
    if (A.tp3) {
        A.tp_q.step(c, bq, nq);
        A.tp_k.step(c, self ? bq : bkv, nk);
    } else {
        RAD_ISSUE_N(c, A.op_qn, nq, q, RAD_W(A.w_qn), q);
        RAD_ISSUE_N(c, A.op_kn, nk, k, RAD_W(A.w_kn), k);
    }
