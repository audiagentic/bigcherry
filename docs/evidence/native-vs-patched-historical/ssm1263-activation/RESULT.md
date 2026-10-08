# 1263 (PRBE41 ssm_conv channels-major) on dual 7900 XTX `-sm tensor`: CRASH at startup

Build 94f23cb9/669acfc2 (experiment ssm-conv-channels-major, gfx1100, linux-multi, b11233). Patch applies cleanly (patch-rebase-check CLEAN)
and compiles, but `llama-server -m Qwen3.8-27B-Q8_0 -sm tensor` aborts during the first graph allocation:

    ggml-backend-meta.cpp:647: GGML_ASSERT(src_ss[0].nr[0] == 1) failed
    ggml_backend_meta_get_split_state -> ggml_backend_meta_buffer_init_tensor -> ggml_gallocr_alloc_graph

The tensor-parallel meta backend cannot derive a split state for the reshape 1263 introduces in the GDN conv path (transpose removed,
channels-major view). Not reached: activation marker (0 hits), any A/B. Verdict for this lane: BLOCKED / incompatible with `-sm tensor`.
Would need the patch to keep the split-state-compatible layout, or a meta-backend fix; layer-split would not exercise the production path.
