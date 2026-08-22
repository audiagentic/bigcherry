// HI82 design item 6: pure-function unit test for 1005_prompt_cache_checkpoint_
// selection's server_prompt_cache_n_reusable(), extracted VERBATIM (via
// patches/1005_prompt_cache_checkpoint_selection.py's own Edit.text, not
// retyped) from the patch this session validated by inspection only --
// no live server test was possible because it requires a real client
// request sequence hitting the *same* slot across multiple turns with a
// stale/rolled-back hybrid-memory cache entry, which none of the models
// or benches used elsewhere this session exercise. This closes the
// "unit-test the pure checkpoint-selection function with synthetic
// checkpoint metadata" half of GPT's item 6 design without needing a
// live model integration test.
//
// Minimal stand-ins for server_prompt_cache_state's shape (only the two
// members server_prompt_cache_n_reusable actually touches).
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <vector>

struct checkpoint_t { int64_t n_tokens; };
struct prompt_t {
    std::vector<checkpoint_t> checkpoints;
    struct tokens_t { size_t n; size_t size() const { return n; } } tokens;
};
struct server_prompt_cache_state { prompt_t prompt; };

#include "n_reusable_verbatim.inc"

static int failures = 0;

static void expect(const char *name, int64_t got, int64_t want) {
    if (got != want) {
        printf("FAIL %-45s got=%lld want=%lld\n", name, (long long)got, (long long)want);
        failures++;
    } else {
        printf("ok   %-45s = %lld\n", name, (long long)got);
    }
}

int main() {
    // 1. Fully-contained cached prompt (lcp == cached size): everything is
    //    reusable regardless of checkpoints, since nothing needs rollback.
    {
        server_prompt_cache_state s;
        s.prompt.tokens.n = 100;
        s.prompt.checkpoints = {{50}};
        expect("fully_contained_ignores_checkpoints", server_prompt_cache_n_reusable(s, 100), 100);
    }

    // 2. No checkpoints at all (ordinary-transformer / SWA-only state):
    //    rollback-capable, so the entire common prefix is reusable.
    {
        server_prompt_cache_state s;
        s.prompt.tokens.n = 500;
        s.prompt.checkpoints = {};
        expect("no_checkpoints_full_lcp_reusable", server_prompt_cache_n_reusable(s, 200), 200);
    }

    // 3. Recurrent-state-compatible: exact checkpoint sits AT the divergence
    //    point (n_tokens == lcp) -- fully usable.
    {
        server_prompt_cache_state s;
        s.prompt.tokens.n = 1000;
        s.prompt.checkpoints = {{300}};
        expect("checkpoint_exactly_at_lcp", server_prompt_cache_n_reusable(s, 300), 300);
    }

    // 4. Stale checkpoint: only a checkpoint PAST the divergence point
    //    exists (n_tokens > lcp) -- nothing at or below lcp is usable, must
    //    fall back to zero (reprocess from scratch), not the stale value.
    {
        server_prompt_cache_state s;
        s.prompt.tokens.n = 1000;
        s.prompt.checkpoints = {{900}};
        expect("stale_checkpoint_past_lcp_yields_zero", server_prompt_cache_n_reusable(s, 300), 0);
    }

    // 5. Post-divergence candidate set: multiple checkpoints straddling lcp
    //    -- must pick the LARGEST one still <= lcp, not the first or last.
    {
        server_prompt_cache_state s;
        s.prompt.tokens.n = 1000;
        s.prompt.checkpoints = {{100}, {250}, {600}, {900}};
        expect("largest_checkpoint_at_or_below_lcp", server_prompt_cache_n_reusable(s, 300), 250);
    }

    // 6. Empty checkpoint at position 0 only, lcp beyond it and beyond cache
    //    size but not fully-contained (lcp < cached size): only the 0-valued
    //    checkpoint qualifies -- reusable salvage is legitimately zero, but
    //    via the checkpoint-scan branch, not the fully-contained short-circuit.
    {
        server_prompt_cache_state s;
        s.prompt.tokens.n = 1000;
        s.prompt.checkpoints = {{0}};
        expect("only_zero_checkpoint_below_lcp", server_prompt_cache_n_reusable(s, 300), 0);
    }

    printf(failures == 0 ? "\nALL PASSED\n" : "\n%d FAILURE(S)\n", failures);
    return failures == 0 ? 0 : 1;
}
