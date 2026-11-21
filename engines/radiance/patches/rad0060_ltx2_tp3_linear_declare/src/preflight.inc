        /* BC_LTX3_C01_PREFLIGHT: a diagnostic builder seam, never serving admission. */
        if (slice && (!g.w_i8h || g.w_i8_group != RAD_I8_GROUP_WIDE || g.w_bf16 || small ||
                      !slice->valid(N, K, g.w_i8_group) ||
                      (slice->mode == C01Slice::Mode::Rows &&
                       (wsrc.size() != (size_t)slice->banks ||
                        bsrc.size() != (size_t)slice->banks ||
                        parts.size() != (size_t)slice->banks ||
                        !std::all_of(parts.begin(), parts.end(), [&](int64_t part) {
                            return part == slice->source_width;
                        }))) ||
                      (slice->mode == C01Slice::Mode::Columns &&
                       (bsrc.size() || parts.size() || wsrc.size() != 1)))) {
            fprintf(stderr, "radiance: ltx2: C01 IQ8R shard invalid, incompatible fused banks, "
                            "or output requires bias-once reduction not yet implemented\n");
            return RAD_E_UNSUPPORTED;
        }
