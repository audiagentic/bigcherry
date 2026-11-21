    /* BC_LTX3_C04_FINALIZE: reuse Radiance's exact three-rank all_reduce.
     * Every rank receives a full reduced replica and applies its full-width
     * BF16 bias once, never to the rank-local partial GEMM.
     * This diagnostic is not an opt-in production architecture graph.
     */
    int finalize_deferred_output(RadBuilder* b, Names& nm, const Geom& g,
                                 const char* base, int access, int group,
                                 rad_buf y, int64_t max_rows) {
        if (!deferred_bias || !op || bias || op_b || post_reduce ||
            !y || g.world != 3 || g.wire_exact != 1 ||
            N <= 0 || max_rows <= 0 ||
            max_rows > std::numeric_limits<int64_t>::max() / N)
            return RAD_E_UNSUPPORTED;
        // Lin::declare already registered exactly one checkpoint bias source.
        const char* bn = nm.f("%s.bias", base);
        bias = decl_w(b, bn, RAD_BF16, {N}, access,
                      RAD_SHARD_NONE, grp_layer(group));
        if (!bias) return RAD_E_INVAL;
        post_reduce = rw(b, RAD_OP(b, "all_reduce",
                                  RAD_PARAMS(RAD_INT("world_size", g.world),
                                             RAD_RANGE("numel", N, max_rows * N),
                                             RAD_STR("dtype", "bf16"),
                                             RAD_INT("exact", g.wire_exact),
                                             RAD_INT("min_bytes", g.wire_min_bytes)),
                                  RAD_NOWEIGHTS),
                         {y}, {y});
        if (!post_reduce) return RAD_E_UNSUPPORTED;
        op_b = rw(b, RAD_OP(b, "table_add",
                            RAD_PARAMS(RAD_RANGE("M", 1, max_rows),
                                       RAD_INT("n", N), RAD_INT("L", 1),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(bias)),
                  {y}, {y});
        return op_b ? RAD_OK : RAD_E_UNSUPPORTED;
    }
