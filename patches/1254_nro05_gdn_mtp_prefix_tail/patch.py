"""NRO05 draft: eligibility/prefix helper for GDN MTP prefix chunking."""

from bigcherry.patcher import Edit, FilePatch

DRAFT_SOURCE_CONTEXT = {
    "parent": "NRO04",
    "source-commit": "4169fbbf50d24beb6d269a2350e7f780b85369e6",
    "scope": "K>1 prefix chunking; final K snapshot tokens stay sequential",
}

_TEXT = r'''
// BIGCHERRY_NRO05_GDN_MTP_PREFIX_SCAFFOLD_BEGIN
static bool bigcherry_nro05_mtp_prefix_candidate(
        bool kda, int K, int64_t n_tokens, int64_t n_seqs, int64_t S_v, int64_t * n_prefix) {
    if (kda || K <= 1 || n_seqs != 1 || n_tokens <= (int64_t) K + 64 ||
            (S_v != 16 && S_v != 32 && S_v != 64 && S_v != 128)) {
        return false;
    }
    *n_prefix = n_tokens - K;
    return *n_prefix > 0;
}
// Runtime routing intentionally remains stock until snapshot-slot fixtures pass.
// BIGCHERRY_NRO05_GDN_MTP_PREFIX_SCAFFOLD_END

'''

PATCHES = [FilePatch(
    path="ggml/src/ggml-cuda/gated_delta_net.cu",
    description="stage fail-closed MTP GDN prefix/tail eligibility",
    edits=(Edit(
        id="mtp-prefix-helper",
        anchor=r"^namespace bigcherry_rd50_gdn_chunked \{$",
        mode="insert_before",
        text=_TEXT,
        guard=r"BIGCHERRY_NRO05_GDN_MTP_PREFIX_SCAFFOLD_BEGIN",
        rationale="insert NRO05 adjacent to the RD50/NRO04 chunked source unit using the parent namespace code anchor",
    ),),
)]
