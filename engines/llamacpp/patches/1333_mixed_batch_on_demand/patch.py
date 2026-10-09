"""1333: build the mixed token/embd input branch only for a ubatch that is actually mixed.

Upstream #29622 (0bb496dbd, "llama: support both embd + raw tokens in batch") lets one batch carry token ids and
embedding rows. build_inp_embd adds a third select branch for it whenever the architecture supports mixed batches
(all but six) and the context is a default one - so every graph of every served model carries the inputs
mixed_tokens, mixed_slots and mixed_embd (F32 [n_embd, n_tokens], as large as the embd input), a dup of it, a
second token-embedding lookup and a set_rows, although the branch is only selected for a mixed ubatch.

Bisect on Brutus (2x gfx1100 + gfx1201 tensor split, gfx1030 MTP drafter), same patch recipe on every build:
upstream 2ca15f540 (the commit before) is fast, 0bb496dbd is slow - Flash-Next 24K MTP decode 41.1 -> 44.6 ms/step,
Qwen3.8-27B dual-XTX prefill 1290 -> 1267-1276 t/s at 10K and 1247 -> 1228-1230 t/s at 32K. Greedy text and draft
acceptance are identical on both sides.

The graph is built per ubatch with its contents known, and llm_graph_params already compares is_mixed() before a
graph is reused, so the branch is added only when ubatch.is_mixed(): token-only and embd-only ubatches get the
pre-#29622 two-branch graph, a mixed ubatch gets upstream's three-branch graph. No flag; the batch allocator keeps
accepting mixed batches as upstream does (that is a validity check, not a cost).

Trade-off against upstream: the three-branch graph is no longer part of the worst-case reserve, so the first mixed
ubatch of a context changes the graph shape and the scheduler reallocates once.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "core"
STATE = "validated"

_A_GATE = ("    const bool has_mixed = llm_arch_supports_mixed_batch(arch) && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT;\n"
           "    if (has_mixed) {\n")
_N_GATE = ("    // BigCherry 1333: only a mixed ubatch pays for the mixed branch (three graph inputs, a dup, a second token\n"
           "    // lookup); llm_graph_params compares is_mixed(), so a graph is never reused across the two shapes\n"
           "    const bool has_mixed = llm_arch_supports_mixed_batch(arch) && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&\n"
           "        ubatch.is_mixed();\n"
           "    if (has_mixed) {\n")

PATCHES = [
    FilePatch(
        path="src/llama-graph.cpp",
        description="1333: build_inp_embd adds the mixed token/embd branch only for a mixed ubatch",
        language="none",
        edits=(
            Edit(id="mixed-batch-on-demand", anchor=_re.escape(_A_GATE), mode="replace", text=_N_GATE,
                 guard=r"BigCherry 1333: only a mixed ubatch pays for the mixed branch",
                 rationale="The has_mixed decision in llm_graph_context::build_inp_embd, directly before the branch is built.",
                 expect_matches=1, max_span_lines=3),
        ),
    ),
]
