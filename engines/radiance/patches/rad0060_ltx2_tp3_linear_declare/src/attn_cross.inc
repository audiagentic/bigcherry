    else {
        RAD_ARCH_TRY(A.qkv.declare(b, nm, g, nm.f("%s.q", dn), { w("to_q") }, { bs("to_q") }, tp ? std::initializer_list<int64_t>{full} : std::initializer_list<int64_t>{}, in,
                                   qdim, acc, group, xin, bq, rows_q, false, tp)); /* BC_LTX3_LTX03_CROSS_Q */
        const C01Slice kv_slice = C01Slice::rows(full, tp ? tp->offset : 0, in, hd, 2);
        RAD_ARCH_TRY(A.kv.declare(b, nm, g, nm.f("%s.kv", dn), { w("to_k"), w("to_v") },
                                  { bs("to_k"), bs("to_v") }, { full, full }, 2 * in, kvdim, acc, group,
                                  *kvin, bkv, rows_k, false, tp ? &kv_slice : nullptr)); /* BC_LTX3_LTX03_CROSS_KV */
    }
