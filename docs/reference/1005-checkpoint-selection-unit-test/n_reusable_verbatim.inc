// how many tokens of a cached prompt could actually be reused if we restored it
static int64_t server_prompt_cache_n_reusable(const server_prompt_cache_state & state, int lcp) {
    // the cached prompt is fully contained in the new one - nothing has to be
    // rolled back, so all of it is reusable
    if (lcp == (int) state.prompt.tokens.size()) {
        return lcp;
    }

    // no checkpoints - the memory can be rolled back to an arbitrary position,
    // so the entire common prefix is reusable
    if (state.prompt.checkpoints.empty()) {
        return lcp;
    }

    // without a rollback-capable memory module (hybrid/recurrent), the restored
    // state is usable only up to its newest context checkpoint at or below the
    // divergence point - that checkpoint is what the caller falls back to
    // instead of reprocessing the prompt from scratch
    int64_t res = 0;

    for (const auto & ckpt : state.prompt.checkpoints) {
        if (ckpt.n_tokens <= lcp) {
            res = std::max(res, ckpt.n_tokens);
        }
    }

    return res;
}

