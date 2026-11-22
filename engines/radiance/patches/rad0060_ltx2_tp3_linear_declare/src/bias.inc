            /* BC_LTX3_C01_BIAS: fused Q/K/V bias tracks the source bank rows. */
            if (slice && slice->mode == C01Slice::Mode::Rows) {
                if (slice->banks == 3)
                    bias = decl_w(b, bn, RAD_BF16, { N }, access, RAD_SHARD_ROW,
                                  grp_layer(group), 0,
                                  {slice->width, slice->width, slice->width});
                else if (slice->banks == 2)
                    bias = decl_w(b, bn, RAD_BF16, { N }, access, RAD_SHARD_ROW,
                                  grp_layer(group), 0, {slice->width, slice->width});
                else
                    bias = decl_w(b, bn, RAD_BF16, { N }, access, RAD_SHARD_ROW,
                                  grp_layer(group), 0, {slice->width});
                if (!bias) return RAD_E_INVAL;
                const int64_t lo[3] = {slice->offset, slice->offset, slice->offset};
                const int64_t hi[3] = {slice->offset + slice->width,
                                       slice->offset + slice->width,
                                       slice->offset + slice->width};
                const int64_t full[3] = {slice->source_width, slice->source_width,
                                          slice->source_width};
                if (slice->banks == 1)
                    RAD_ARCH_TRY(rad_weight_shard_span(b, bias, lo[0], hi[0]));
                else
                    RAD_ARCH_TRY(rad_weight_shard_span_parts(b, bias, lo, hi, full, slice->banks));
            } else if (slice && slice->defer_bias) {
                // The partial column GEMM must never apply bias before the
                // reduction. Name mapping is retained for a future unique
                // finalizer; no rank-partial bias weight or table_add is declared.
                deferred_bias = true;
            } else {
                bias = decl_w(b, bn, RAD_BF16, { N }, access, RAD_SHARD_NONE, grp_layer(group));
            }
            if (!bias && !deferred_bias) return RAD_E_INVAL;
