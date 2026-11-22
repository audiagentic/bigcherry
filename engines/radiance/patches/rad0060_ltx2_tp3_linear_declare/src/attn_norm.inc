    /* BC_LTX3_C06_ATTN_NORM: unchanged TP1 norm, or the shared
     * C05 full-width normalizer; Q and K sequentially share scratch. */
    if (tp) {
        RAD_ARCH_TRY(A.tp_q.declare(b, g, bq, rms_scratch, A.w_qn,
                                    full, tp->offset, in, 0, hd,
                                    rows_q, eps));
        RAD_ARCH_TRY(A.tp_k.declare(b, g, kb, rms_scratch, A.w_kn,
                                    full, tp->offset, in, in, hd,
                                    rows_k, eps));
    } else {
        A.op_qn = rms_op(b, rows_q, in, eps, A.w_qn, bq, bq);
        A.op_kn = rms_op(b, rows_k, in, eps, A.w_kn, kb, kb);
    }
