        /* BC_LTX3_C01_PREFLIGHT: a diagnostic builder seam, never serving admission. */
        if (slice && (!g.w_i8h || g.w_i8_group != RAD_I8_GROUP_WIDE || g.w_bf16 || small ||
                      !slice->valid(N, K, g.w_i8_group) ||
                      (slice->mode == C01Slice::Mode::Rows &&
                       (wsrc.size() != (size_t)slice->banks ||
                        (slice->banks == 1 ? bsrc.size() > 1 : bsrc.size() != (size_t)slice->banks) ||
                        (slice->banks == 1 ? parts.size() > 1 : parts.size() != (size_t)slice->banks) ||
                        !std::all_of(parts.begin(), parts.end(), [&](int64_t part) {
                            return part == slice->source_width;
                        }))) ||
                      (slice->mode == C01Slice::Mode::Columns &&
                       (parts.size() || wsrc.size() != 1 ||
                        (slice->defer_bias ? bsrc.size() > 1 : bsrc.size() != 0))))) {
            fprintf(stderr, "radiance: ltx2: C01 IQ8R shard invalid, incompatible fused banks, "
                            "or output requires explicit reduction ownership\n");
            return RAD_E_UNSUPPORTED;
        }
        // The full-width output must be reduced even for biasless video FFN.
        if (slice && slice->mode == C01Slice::Mode::Columns && slice->defer_bias) {
            deferred_bias = true;
            deferred_has_bias = bsrc.size() != 0;
        }
