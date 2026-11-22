/* BC_LTX3_C05_GLOBAL_RMS: diagnostic all-head RMS without a new kernel.
 * The runtime already has fill, cast, exact all_reduce and rmsnorm.
 * Pack each local Q/K shard into an all-head scratch, reduce full-width
 * disjoint columns, run the unchanged TP1 RMS, extract owned columns.
 * This is bandwidth-heavy; callers own and may reuse scratch for Q and K.
 * No production attention path calls this diagnostic yet.
 */
struct Tp3GlobalRms {
    rad_buf scratch = 0;
    rad_weight gain = 0;
    rad_op clear = 0, insert = 0, reduce = 0, norm = 0, extract = 0;
    int64_t full = 0, lo = 0, local = 0, col = 0, max_rows = 0;

    int declare(RadBuilder* b, const Geom& g, rad_buf source, rad_buf shared,
                rad_weight full_gain, int64_t all_width, int64_t offset,
                int64_t owned_width, int64_t source_column, int64_t head_dim,
                int64_t rows, double eps) {
        if (!b || !source || !shared || !full_gain || scratch || clear ||
            g.world < 2 || g.wire_exact != 1 || g.wire_min_bytes < 0 ||
            !g.dtype || std::strcmp(g.dtype, "bf16") != 0 ||
            all_width <= 0 || owned_width <= 0 || offset < 0 ||
            offset > all_width || owned_width > all_width - offset ||
            head_dim <= 0 || all_width % head_dim ||
            offset % head_dim || owned_width % head_dim ||
            source_column < 0 ||
            source_column > std::numeric_limits<int64_t>::max() - owned_width ||
            rows <= 0 || rows > std::numeric_limits<int64_t>::max() / all_width ||
            !std::isfinite(eps) || eps <= 0.0)
            return RAD_E_UNSUPPORTED;

        full = all_width;
        lo = offset;
        local = owned_width;
        col = source_column;
        max_rows = rows;
        scratch = shared;
        gain = full_gain;
        clear = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_RANGE("M", 1, rows),
                                        RAD_INT("n", full),
                                        RAD_STR("dtype", "bf16"),
                                        RAD_F64("value", 0.0)),
                             RAD_NOWEIGHTS),
                   {}, {scratch});
        insert = cast_op(b, rows, local, "bf16", "bf16", source, scratch);
        reduce = rw(b, RAD_OP(b, "all_reduce",
                              RAD_PARAMS(RAD_INT("world_size", g.world),
                                         RAD_RANGE("numel", full, rows * full),
                                         RAD_STR("dtype", "bf16"),
                                         RAD_INT("exact", g.wire_exact),
                                         RAD_INT("min_bytes", g.wire_min_bytes)),
                              RAD_NOWEIGHTS),
                    {scratch}, {scratch});
        norm = rms_op(b, rows, full, eps, gain, scratch, scratch);
        extract = cast_op(b, rows, local, "bf16", "bf16", scratch, source);
        return clear && insert && reduce && norm && extract ? RAD_OK : RAD_E_UNSUPPORTED;
    }

    void step(RadCtx* c, rad_buf source, int64_t rows) const {
        if (rows <= 0) return;
        if (!clear || !insert || !reduce || !norm || !extract || !scratch ||
            !gain || !source || rows > max_rows) {
            rad_step_fail(c, "LTX TP3 all-head RMS was not declared");
            return;
        }
        RAD_ISSUE_N(c, clear, rows, brows(scratch, rows));
        RAD_ISSUE_N(c, insert, rows, bcol(source, col, local, rows),
                    bcol(scratch, lo, local, rows));
        RAD_ISSUE_N(c, reduce, rows * full, brows(scratch, rows), RAD_NONE);
        RAD_ISSUE_N(c, norm, rows, brows(scratch, rows),
                    RAD_W(gain), brows(scratch, rows));
        RAD_ISSUE_N(c, extract, rows, bcol(scratch, lo, local, rows),
                    bcol(source, col, local, rows));
    }
};
