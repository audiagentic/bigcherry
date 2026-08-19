"""RD20: align attn_gate tensor-parallel split granularity with attn_q.

Provenance (group 'rdna-boosts' patches are external backports; the
machine-readable PROVENANCE dict below is cross-checked against
external-sources.toml by tools/tests/test_external_sources.py):

  source:            stew675-rdna-boosts
  repo:              https://github.com/stew675/llama.cpp
  locator:           rdna-boosts (branch name is a locator only, NOT identity)
  fork commit:       3b200b259c48a688929cae6bfe1048a60543bc67
                     "llama : align attn_gate TP split granularity with attn_q"
  original commit:   ed89854b2aeb0e333dd61424f14af2aedaca126e
                     (pre-rebase snapshot v1 -- this was the reviewed head of
                     snapshot v1; content-identical to the fork commit above,
                     verified by git patch-id on 2026-08-18)
  reviewed snapshot: v2 -- head 9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22
                     on base 4df29be4f4c3673f428170fda944a5b19f743bb8
  plan item:         RD20 (docs/planning/active/rdna-boost-experiments/RD20.md)
  mainline status:   NOT merged into ggml-org/llama.cpp master as of tip
                     6d0549831 (git cherry patch-id check, 2026-08-18)

What it does (correctness, not performance):
  The regular-attention branch of get_split_granularity (inside
  llama_meta_device_get_split_state) gave attn_gate.weight a granularity of
  1, so it split evenly across devices, while the attention weights it feeds
  from split at granularity_q (head-aligned). With 3+ GPUs the head-aligned
  split can round a device's share to zero (e.g. {0, 2048, 2048} for a
  4096-dim q with n_embd_q = 2048), and the element-wise MUL between the
  attn output and the gate then ABORTS on incompatible split states.

  The fix gives attn_gate.weight the same head-aligned granularity as
  attn_q, matching what the recursive-attention branch (the pattern_qkv
  path) already does. Note: the QWEN3NEXT/QWEN35/QWEN35MOE granularity
  doubling inside the if-branch now applies to attn_gate.weight too, which
  is the intended alignment (the gate is a per-head weight like the q it
  scales).

  This patch changes what a multi-GPU tensor-split run DOES, not which
  candidate the autotune dispatch engine picks; the tuner cannot discover
  or reproduce this itself.

Porting notes:
  - Ported VERBATIM (one hunk, one condition extended) from the fork commit.
    The anchor region is byte-identical on the pinned llama.cpp
    4801e3c567d5131dd41b387df5f2d4b1370d92be (llama-model.cpp,
    llama_meta_device_get_split_state, ~line 633).
  - pattern_attn_gate_weight is a file-static regex already declared at
    line 367 of the same function on the pin -- no new symbols.
  - No dependency on any other BigCherry patch; no BigCherry patch touches
    src/llama-model.cpp's split-granularity region.
  - Interacts with RD19 (1200): RD19 removes the Meta wrapper entirely for
    single-GPU tensor split, so RD20's code path is only reachable with
    2+ devices. They are independent hunks in independent files and can be
    tested in either order; the isolated bench for RD20 needs 2+ GPUs
    (Brutus devices 0/1, both gfx1100) to exercise the Meta path at all.

Isolation and promotion (first-sweep policy, RD review 2026-08-18):
  - GROUP 'rdna-boosts' + STATE 'untested' keeps this OUT of the production
    'framework' and 'validated-enhancements' patch-sets. It is built ONLY
    when explicitly selected:
        python -m bigcherry apply --groups core,rdna-boosts \
                                  --states validated,untested
  - Validation is correctness-first (RD20's own acceptance): output
    equality vs native across 1/2/3+ GPU layouts where available; the
    abort-on-incompatible-split-states is the bug, so the pre-patch
    reproduction (3-GPU run, or the {0, n, n} zero-share geometry) is the
    regression test.
  - On promotion: set STATE = 'validated' and add this patch id to
    [patch-set.validated-enhancements] in recipes.toml.

Maintenance (future pin bumps / fork movement):
  - If the get_split_granularity lambda changes shape upstream, re-derive
    the hunk from the tracked fork commit (or its successor) in
    external-sources.toml -- do not reconstruct from memory.
  - Run `python -m bigcherry sources check` to detect: (a) the fork branch
    moving or being rebased again, (b) this change landing in mainline
    (patch-id match), which would make this patch redundant on the next
    pin bump, (c) content drift of the fork commit vs the original.
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

PROVENANCE = {
    "source-id": "stew675-rdna-boosts",
    "plan-item": "RD20",
    "fork-commit": "3b200b259c48a688929cae6bfe1048a60543bc67",
    "fork-commit-title": "llama : align attn_gate TP split granularity with attn_q",
    "original-commit": "ed89854b2aeb0e333dd61424f14af2aedaca126e",
    "snapshot-head": "9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22",
    "snapshot-base": "4df29be4f4c3673f428170fda944a5b19f743bb8",
    "adaptations": [],
}


_OLD = """            const int64_t granularity_q = std::lcm(n_embd_q, blck_size_perf);
            if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_q_bias)) {"""

_NEW = """            const int64_t granularity_q = std::lcm(n_embd_q, blck_size_perf);
            if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_q_bias) ||
                    std::regex_match(tensor_name, pattern_attn_gate_weight)) {"""


PATCH = FilePatch(
    path="src/llama-model.cpp",
    description="Give attn_gate.weight the same tensor-parallel split "
                "granularity as attn_q (rdna-boosts 3b200b259 / RD20)",
    edits=(
        Edit(
            id="rd20-attn-gate-granularity",
            anchor=re.escape(_OLD),
            rationale="llama_meta_device_get_split_state -> "
                      "get_split_granularity lambda: the regular-attention "
                      "head-aligned branch, which currently covers only "
                      "attn_q weight/bias",
            mode="replace",
            text=_NEW,
            # NOTE: the guard must include the `||` continuation: the file
            # already contains an unrelated single-match line
            # `if (std::regex_match(tensor_name, pattern_attn_gate_weight)) {`
            # (tensor mapping, ~line 469) that a looser guard would match and
            # falsely report as already-applied.
            guard=(r"pattern_q_bias\) \|\|\s+"
                   r"std::regex_match\(tensor_name, pattern_attn_gate_weight\)\) \{"),
        ),
    ),
)

PATCHES = [PATCH]
