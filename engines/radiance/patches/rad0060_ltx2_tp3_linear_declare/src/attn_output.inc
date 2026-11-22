    /* BC_LTX3_C06_ATTN_OUTPUT: one existing local IQ8R GEMM,
     * existing exact collective, full BF16 bias once on every replica. */
    const C01Slice output_slice = C01Slice::columns_deferred(
        full, tp ? tp->offset : 0, in);
    RAD_ARCH_TRY(A.out.declare(b, nm, g, nm.f("%s.out", dn), { w("to_out.0") },
                               { bs("to_out.0") }, {}, odim, in, acc, group,
                               ao, y, rows_q, false, tp ? &output_slice : nullptr));
    if (tp)
        RAD_ARCH_TRY(A.out.finalize_deferred_output(
            b, nm, g, nm.f("%s.out", dn), acc, group, y, rows_q));
