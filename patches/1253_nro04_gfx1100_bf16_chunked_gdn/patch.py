"""NRO04 draft: gfx1100 BF16/WMMA GDN eligibility scaffold.

The large WMMA kernel remains deliberately absent until fragment-layout tests
exist.  This patch therefore cannot change runtime output.
"""

from bigcherry.patcher import Edit, FilePatch

DRAFT_SOURCE_CONTEXT = {
    "repo": "https://github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt",
    "commit": "4169fbbf50d24beb6d269a2350e7f780b85369e6",
    "source-file": "ggml/src/ggml-cuda/gated_delta_net_chunked_bf16_gfx11.cu",
    "pin": "b10705",
}

_TEXT = r'''
// BIGCHERRY_NRO04_GFX1100_BF16_GDN_SCAFFOLD_BEGIN
static bool bigcherry_nro04_gfx1100_bf16_candidate(
        int cc, int64_t S_v, bool kda, int K, int64_t n_tokens) {
    const char * env = getenv("GGML_CUDA_GDN_CHUNKED_BF16_GFX1100");
    const bool requested = env != nullptr && atoi(env) != 0;
    return requested && GGML_CUDA_CC_IS_RDNA3(cc) && S_v == 128 && !kda && K == 1 && n_tokens > 64;
}

// Runtime dispatch intentionally remains disabled until the gfx1100 WMMA
// fragment probe described by NRO04 is committed and passing.
static bool bigcherry_nro04_gfx1100_bf16_ready() {
    return false;
}
// BIGCHERRY_NRO04_GFX1100_BF16_GDN_SCAFFOLD_END

'''

PATCHES = [FilePatch(
    path="ggml/src/ggml-cuda/gated_delta_net.cu",
    description="stage gfx1100 BF16/WMMA GDN eligibility without enabling unvalidated device code",
    edits=(Edit(
        id="gfx1100-bf16-policy-scaffold",
        anchor=r"^namespace bigcherry_rd50_gdn_chunked \{$",
        mode="insert_before",
        text=_TEXT,
        guard=r"BIGCHERRY_NRO04_GFX1100_BF16_GDN_SCAFFOLD_BEGIN",
        rationale="NRO04 extends the RD50 chunked-GDN unit and must apply after RD50",
    ),),
)]
