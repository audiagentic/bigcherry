    /* BC_LTX3_C06_ATTN_GEOMETRY: exact Q/K and output reductions for
     * self or cross attention. Refuse missing independent KV before mutation. */
    if (tp && (!rms_scratch || !xin.q.op || !ao.q.op ||
               (!self && (!kvin || !kvin->q.op || !kvin->a.q8_fed)) ||
               !xin.a.q8_fed || !ao.a.q8_fed ||
               g.world != 3 || g.wire_exact != 1 ||
               H <= 0 || hd <= 0 || H > std::numeric_limits<int64_t>::max() / hd ||
               tp->mode != C01Slice::Mode::Rows || tp->banks != (self ? 3 : 1) ||
               tp->source_width != H * hd || tp->head_dim != hd ||
               tp->width <= 0 || tp->offset < 0 ||
               tp->offset > H * hd || tp->width > H * hd - tp->offset ||
               tp->offset % hd || tp->width % hd ||
               qdim <= 0 || kvdim <= 0 || odim <= 0 ||
               g.w_i8_group <= 0 ||
               qdim % g.w_i8_group || kvdim % g.w_i8_group ||
               ((self) && (qdim != H * hd || kvdim != H * hd ||
                            odim != H * hd)) ||
               rows_q <= 0 || rows_k <= 0 ||
               (self && rows_q != rows_k)))
        return RAD_E_UNSUPPORTED;
    A.tp3 = tp != nullptr;
    A.head0 = tp ? tp->offset / hd : 0;
    A.H = tp ? tp->width / hd : H;
    A.hd = hd;
    A.inner = A.H * hd;
    const int64_t in = A.inner, full = H * hd;
