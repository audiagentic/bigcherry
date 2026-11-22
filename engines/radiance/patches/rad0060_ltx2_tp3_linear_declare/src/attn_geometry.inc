    /* BC_LTX3_C06_ATTN_GEOMETRY: TP3 only for a complete self-attention
     * with all-head norm scratch owned by the caller. Fail before mutation. */
    if (tp && (!self || !rms_scratch || !xin.q.op || !ao.q.op ||
               !xin.a.q8_fed || !ao.a.q8_fed ||
               g.world != 3 || g.wire_exact != 1 ||
               H <= 0 || hd <= 0 || H > std::numeric_limits<int64_t>::max() / hd ||
               tp->mode != C01Slice::Mode::Rows || tp->banks != 3 ||
               tp->source_width != H * hd || tp->head_dim != hd ||
               tp->width <= 0 || tp->offset < 0 ||
               tp->offset > H * hd || tp->width > H * hd - tp->offset ||
               tp->offset % hd || tp->width % hd ||
               qdim != H * hd || kvdim != H * hd ||
               odim != H * hd || rows_q != rows_k))
        return RAD_E_UNSUPPORTED;
    A.tp3 = tp != nullptr;
    A.head0 = tp ? tp->offset / hd : 0;
    A.H = tp ? tp->width / hd : H;
    A.hd = hd;
    A.inner = A.H * hd;
    const int64_t in = A.inner, full = H * hd;
