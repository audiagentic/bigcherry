"""1333: build the mixed token/embd input branch only on request (BIGCHERRY_MIXED_BATCH=1).

Upstream #29622 (0bb496dbd, "llama: support both embd + raw tokens in batch") lets one batch carry token ids and
embedding rows. It gates the feature on llm_arch_supports_mixed_batch(arch), which is true for every architecture
except six, so every default context now builds a third input-embedding branch into each graph: the inputs
mixed_tokens, mixed_slots and mixed_embd (F32 [n_embd, n_tokens], as large as the embd input), a second token
embedding lookup and a set_rows, and the batch allocator runs in mixed-capable mode. The branch is never selected
unless a batch really mixes the two kinds, but its inputs are part of every graph.

Bisect on Brutus (2x gfx1100 + gfx1201 tensor split, gfx1030 MTP drafter), same patch recipe on every build:
upstream 2ca15f540 (the commit before) is fast, 0bb496dbd is slow - Flash-Next 24K MTP decode 41.1 -> 44.6 ms/step,
and b11401/b11402 (which contain it) lose 0.8-1.5% of Qwen3.8-27B dual-XTX prefill. Greedy text and draft
acceptance are identical on both sides.

Nothing BigCherry serves sends a mixed batch (MTP hands the hidden state over as an embd-only batch), so the
predicate returns false unless BIGCHERRY_MIXED_BATCH=1: the graph and the batch allocator are then exactly the
pre-#29622 shape. With the flag set the upstream behaviour is unchanged. A mixed batch without the flag is rejected
by the batch allocator as before #29622 ("batch invalid").
"""

import re as _re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "core"
STATE = "untested"

_A_INC = "#include <map>\n"
_N_INC = "#include <cstdlib>\n#include <map>\n"

_A_FN = "bool llm_arch_supports_mixed_batch(const llm_arch & arch) {\n    switch (arch) {\n"
_N_FN = """bool llm_arch_supports_mixed_batch(const llm_arch & arch) {
    // BigCherry 1333: the mixed token/embd input branch (three extra graph inputs per ubatch) is built only on
    // request; no served workload sends a mixed batch
    static const bool bc_mixed_batch = [] {
        const char * e = getenv("BIGCHERRY_MIXED_BATCH");
        return e != nullptr && e[0] == '1';
    }();
    if (!bc_mixed_batch) {
        return false;
    }
    switch (arch) {
"""

PATCHES = [
    FilePatch(
        path="src/llama-arch.cpp",
        description="1333: llm_arch_supports_mixed_batch is opt-in (BIGCHERRY_MIXED_BATCH=1)",
        language="none",
        edits=(
            Edit(id="mixed-batch-include", anchor=_re.escape(_A_INC), mode="replace", text=_N_INC,
                 guard=r"#include <cstdlib>\n#include <map>\n", rationale="Standard include block of llama-arch.cpp.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="mixed-batch-opt-in", anchor=_re.escape(_A_FN), mode="replace", text=_N_FN,
                 guard=r"BigCherry 1333: the mixed token/embd input branch",
                 rationale="The single predicate both the context constructor and build_inp_embd consult.",
                 expect_matches=1, max_span_lines=3),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_MIXED_BATCH', '0|1', '0 (off)',
           'build the upstream mixed token/embd input branch into every graph (needed only for batches that mix '
           'token ids and embedding rows)'),
)
