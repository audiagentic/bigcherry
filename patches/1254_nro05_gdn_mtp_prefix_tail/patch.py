"""PNRO05 (NRO05): chunk the GDN MTP prefill prefix, sequential snapshot tail.

From nasone 4169fbbf (block 02), on top of 1253: for MTP (K > 1), a single
long sequence (n_tokens > K + 64) runs the BF16 chunked kernel on the first
n_tokens - K tokens into a scratch state, then the sequential kernel on the
last K tokens so the K snapshot slots stay exact. RDNA3/RDNA4, S_v == 128
only (the fork's fp32 fallback is not ported). Opt out with
GGML_CUDA_GDN_CHUNKED=0 / GGML_CUDA_GDN_CHUNKED_BF16=0.

Activation: BIGCHERRY_PATCH_TRACE=1 logs
  BIGCHERRY_PATCH_HIT patch=1254_nro05 path=gdn_mtp_prefix_bf16
"""

from bigcherry.patcher import Edit, FilePatch

PROVENANCE = {
    "source-id": "nasone-rdna-optimizations",
    "plan-item": "NRO05",
    "fork-commit": "4169fbbf50d24beb6d269a2350e7f780b85369e6",
    "port-mode": "port_diff-generated MTP prefix dispatch, BF16 only",
}

GDN_MTP_PREFIX = FilePatch(
    path='ggml/src/ggml-cuda/gated_delta_net.cu',
    description='MTP prefix: BF16 chunked GDN on n_tokens-K, sequential on the last K',
    edits=(
        Edit(
            id='nro05-mtp-01',
            anchor='\\ \\ \\ \\ \\}\\\n\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ float\\ \\*\\ state_d\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ =\\ dst_d\\ \\+\\ S_v\\ \\*\\ H\\ \\*\\ n_tokens\\ \\*\\ n_seqs;\\\n',
            text='    }\n\n    // MTP (K > 1) needs snapshot slots, so the K==1 fused-cache chunked path\n    // cannot cover it. Long single-sequence prefill: chunked GDN on the prefix\n    // (n_tokens - K), sequential GDN only on the last K tokens so slots 0..K-1\n    // stay correct. n_seqs > 1 stays fully sequential — the chunked n_tokens\n    // override is also the sequence stride. Opt out with GGML_CUDA_GDN_CHUNKED=0.\n    if (!kda && K > 1 && n_seqs == 1 && n_tokens > (int64_t) K + 64 &&\n        (S_v == 16 || S_v == 32 || S_v == 64 || S_v == 128)) {\n        const char * env = getenv("GGML_CUDA_GDN_CHUNKED");\n        if (env == nullptr || strcmp(env, "0") != 0) {\n            const int64_t n_prefix = n_tokens - K;\n            ggml_cuda_pool_alloc<float> prefix_state(ctx.pool(), (size_t) H * S_v * S_v * n_seqs);\n            bool prefix_ok = false;\n#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)\n            const int cc_p = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;\n            const bool bf16_rdna_p = GGML_CUDA_CC_IS_RDNA4(cc_p) || GGML_CUDA_CC_IS_RDNA3(cc_p);\n            const char * envb_p = getenv("GGML_CUDA_GDN_CHUNKED_BF16");\n            const bool want_bf16_p = bf16_rdna_p && S_v == 128 &&\n                (envb_p == nullptr || strcmp(envb_p, "0") != 0);\n            if (want_bf16_p) {\n                if (GGML_CUDA_CC_IS_RDNA4(cc_p)) {\n                    prefix_ok = ggml_cuda_op_gated_delta_net_chunked_bf16(ctx, dst, prefix_state.get(), n_prefix);\n                } else {\n                    prefix_ok = ggml_cuda_op_gated_delta_net_chunked_bf16_gfx11(ctx, dst, prefix_state.get(), n_prefix);\n                }\n            }\n#endif\n            if (prefix_ok) {\n                if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {\n                    static std::once_flag bigcherry_gdn_mtp_prefix_bf16_logged;\n                    std::call_once(bigcherry_gdn_mtp_prefix_bf16_logged, [] {\n                        GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1254_nro05 path=gdn_mtp_prefix_bf16\\n");\n                    });\n                }\n                static bool logged_prefix = false;\n                if (!logged_prefix) {\n                    GGML_LOG_INFO("%s: MTP chunked GDN prefix n=%ld K=%d prefix=%ld\\n",\n                                  __func__, (long) n_tokens, K, (long) n_prefix);\n                    logged_prefix = true;\n                }\n                float * state_d = cache ? cache->data : (dst_d + S_v * H * n_tokens * n_seqs);\n                const int64_t state_slot_stride = cache ? cache->slot_stride : (S_v * S_v * H * n_seqs);\n                launch_gated_delta_net<false, true>(\n                    q_d + n_prefix * sq2, k_d + n_prefix * sq2, v_d + n_prefix * sv2,\n                    g_d + n_prefix * sb2, b_d + n_prefix * sb2, prefix_state.get(),\n                    dst_d + n_prefix * S_v * H, state_d,\n                    S_v, H, K, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,\n                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);\n                return;\n            }\n        }\n    }\n\n    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing\n    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;\n',
            mode='replace',
            guard='//\\ MTP\\ \\(K\\ >\\ 1\\)\\ needs\\ snapshot\\ slots,\\ so\\ the\\ K==1\\ fused\\-cache\\ chunked\\ path',
            rationale='nro05-mtp hunk 1: upstream lines 353-352 -> result lines 353-404',
            max_span_lines=6,
        ),
    ),
)

PATCHES = [GDN_MTP_PREFIX]
