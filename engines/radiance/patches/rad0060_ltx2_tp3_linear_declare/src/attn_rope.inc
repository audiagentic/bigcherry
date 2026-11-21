        auto rop = [&](int64_t rows, rad_buf x) {
            /* BC_LTX3_C06_ATTN_ROPE: native rope_split already accepts
             * head0; TP1 retains exactly its original op declaration. */
            if (tp)
                return rw(b, RAD_OP(b, "rope_split",
                                    RAD_PARAMS(RAD_RANGE("M", 1, rows),
                                               RAD_INT("n_head", A.H),
                                               RAD_INT("head_dim", hd),
                                               RAD_INT("dim", rope.dim),
                                               RAD_INT("n_dims", rope.n_dims),
                                               RAD_F64("theta", 10000.0),
                                               RAD_STR("max", rope.max),
                                               RAD_INT("head0", A.head0),
                                               RAD_STR("dtype", "bf16")),
                                    RAD_NOWEIGHTS), {x}, {x});
            return rw(b, RAD_OP(b, "rope_split",
                                RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n_head", H),
                                           RAD_INT("head_dim", hd), RAD_INT("dim", rope.dim),
                                           RAD_INT("n_dims", rope.n_dims), RAD_F64("theta", 10000.0),
                                           RAD_STR("max", rope.max), RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { x }, { x });
        };
