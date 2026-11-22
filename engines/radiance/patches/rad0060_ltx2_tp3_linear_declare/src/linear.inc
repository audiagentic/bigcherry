        /* BC_LTX3_C01_LINEAR: use real LinearFP8, IQ8R logical views.
         * Row part spans select each bank independently; scale,scale.1 follows
         * the code weight's logical row spans. Output columns use whole
         * 256-K scale groups, and no per-rank bias in this initial slice.
         */
        if (!slice) {
            RAD_ARCH_TRY(l.declare(b, nm, g, base, N, K, RAD_SHARD_NONE, grp_layer(group), wsrc, in.a, y,
                                   max_rows, parts));
        } else if (slice->mode == C01Slice::Mode::Rows) {
            if (slice->banks == 3) {
                RAD_ARCH_TRY(l.declare(b, nm, g, base, N, K, RAD_SHARD_ROW, grp_layer(group),
                                       wsrc, in.a, y, max_rows,
                                       {slice->width, slice->width, slice->width}));
            } else if (slice->banks == 2) {
                RAD_ARCH_TRY(l.declare(b, nm, g, base, N, K, RAD_SHARD_ROW, grp_layer(group),
                                       wsrc, in.a, y, max_rows, {slice->width, slice->width}));
            } else {
                RAD_ARCH_TRY(l.declare(b, nm, g, base, N, K, RAD_SHARD_ROW, grp_layer(group),
                                       wsrc, in.a, y, max_rows, {slice->width}));
            }
            const int64_t lo[3] = {slice->offset, slice->offset, slice->offset};
            const int64_t hi[3] = {slice->offset + slice->width,
                                   slice->offset + slice->width,
                                   slice->offset + slice->width};
            const int64_t full[3] = {slice->source_width, slice->source_width, slice->source_width};
            if (slice->banks == 1) {
                RAD_ARCH_TRY(rad_weight_shard_span(b, l.w, lo[0], hi[0]));
                if (l.wref) RAD_ARCH_TRY(rad_weight_shard_span(b, l.wref, lo[0], hi[0]));
                if (l.ws) {
                    if (l.ws_row_div != 1) return RAD_E_UNSUPPORTED;
                    RAD_ARCH_TRY(rad_weight_shard_span(b, l.ws, lo[0], hi[0]));
                }
            } else {
                RAD_ARCH_TRY(l.span_parts(b, lo, hi, full, slice->banks));
            }
        } else {
            RAD_ARCH_TRY(l.declare(b, nm, g, base, N, K, RAD_SHARD_COL, grp_layer(group),
                                   wsrc, in.a, y, max_rows, {}, slice->offset,
                                   slice->offset + slice->width));
        }
