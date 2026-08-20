"""RD44 (AMD-STREAM-006): default the CUDA/HIP graph-optimization pass
(GGML_CUDA_GRAPH_OPT, gated behind an env var previously) to enabled on
RDNA3.5 (gfx1151), keeping the env var as an explicit override on every
architecture.

Provenance (group 'rdna-boosts' patches are external backports; the
machine-readable PROVENANCE dict below is cross-checked against
external-sources.toml by tools/tests/test_external_sources.py):

  source:            amd-ecosystem-llama-cpp
  repo:              https://github.com/AMD-Ecosystem/llama.cpp
  locator:           PR #56 (fork tracks per-PR, not per-branch -- see the
                     source's own registry notes)
  merge commit:      6e9f948a0d8b99db2bce0a83fc021efc2064ef9b
  title:             "ggml-cuda: enable GGML_CUDA_GRAPH_OPT by default on
                     RDNA3.5"
  reviewed:          2026-08-20 (RD28-RD53 validation pass); merged, not
                     ancestral to ggml-org/llama.cpp mainline (fork-only
                     work) and not ancestral to our b10502 pin.

What it does (policy/default flip, not a kernel change -- last item in
the AMD-STREAM chain, RD39->RD40/RD41->RD42->RD43->RD44):
  `ggml_backend_cuda_graph_optimize` previously required
  `GGML_CUDA_GRAPH_OPT=1` to enable the whole concurrency machinery
  (RD39-RD43). This defaults it to enabled on gfx1151 specifically
  (`GGML_CUDA_CC_IS_RDNA3_5`), while every other architecture keeps the
  previous behavior (off unless the env var is set to 1). The env var
  stays an explicit override in both directions:
  `GGML_CUDA_GRAPH_OPT=0` disables it even on gfx1151;
  `GGML_CUDA_GRAPH_OPT=1` forces it on anywhere. The device compute
  capability is only queried when the env var is unset, so the override
  path avoids the extra lookup.

  Upstream-fork measured (gfx1151, Qwen3.6-35B-A3B UD-Q4_K_M): tg128
  +7.3%; VLM decode +8.7%; small-model (Qwen3.5-0.8B) prefill/decode
  within +/-0.4% (noise-level, no regression).

Porting notes:
  - The anchor (the `enable_graph_optimization` lazily-initialized static
    lambda at the top of `ggml_backend_cuda_graph_optimize`) is
    BYTE-IDENTICAL between the fork's PR base and our pinned b10502 tree,
    and untouched by patches 1215 (RD39-RD42, which edits the fan_out loop
    body further down in the same function) and 1216 (RD43, which edits a
    different function entirely) -- verified 2026-08-20 by direct
    comparison against vendor/llama.cpp. Order-independent relative to
    1215/1216; listed after them only to keep the AMD-STREAM chain's
    patch numbers in dependency order.
  - This patch alone does NOT make graph-opt safe to enable: it depends
    on RD39-RD43 (patches 1215+1216) already being applied to actually be
    correct once triggered. Per the plan item's own acceptance criteria,
    promotion needs "hardware-wide confidence" from a broad gfx1151
    regression suite, not just this default flip landing cleanly.

Hardware status (this patch's own acceptance, from the plan item):
  Needs gfx1151 hardware, which Brutus does not have (gfx1100/gfx1201/
  gfx1030 only) -- same hardware gap as RD21/RD22 (patches 1208/1209).
  The patch is behavior-neutral on all three Brutus architectures (the
  RDNA3_5 branch cannot resolve `GGML_CUDA_CC_IS_RDNA3_5`), so the bench
  reduces to a no-regression check on gfx1100/gfx1201/gfx1030; the
  performance claim stays unvalidated until gfx1151 hardware exists.
  UNVALIDATED on BigCherry hardware as of porting -- STATE stays
  "untested".

Isolation and promotion (first-sweep policy, matching existing RD items):
  - GROUP 'rdna-boosts' + STATE 'untested' keeps this OUT of the
    production 'framework' and 'validated-enhancements' patch-sets.
  - Promotion is gated on gfx1151 evidence (same discipline as RD21/RD22,
    patches 1208/1209) plus RD39-RD43 already being validated and
    retained.
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

PROVENANCE = {
    "source-id": "amd-ecosystem-llama-cpp",
    "plan-item": "RD44",
    "fork-commit": "6e9f948a0d8b99db2bce0a83fc021efc2064ef9b",
    "fork-commit-title": "ggml-cuda: enable GGML_CUDA_GRAPH_OPT by default on RDNA3.5",
    "snapshot-head": "58ab0a5f2ce3f426d657d55647846b03fbc1a20b",
    "snapshot-base": "58ab0a5f2ce3f426d657d55647846b03fbc1a20b",
    "adaptations": [
        "None -- the target region is byte-identical between the fork's "
        "PR #56 base and our b10502 pin, and untouched by patches 1215/1216.",
    ],
}

# The anchor stops matching literal text right after "getenv(" and resumes
# after the closing ")": the argument is a string literal ("GGML_CUDA_
# GRAPH_OPT", quotes included), which strip_noise blanks to spaces before
# anchor matching (see csource.strip_noise) -- a literal anchor spanning it
# would never match.
_ENABLE_HEAD = """    static bool enable_graph_optimization = [] {
        const char * env     = getenv("""

_ENABLE_TAIL = """);
        return env != nullptr && atoi(env) == 1;
    }();
"""

_ENABLE_NEW = """    static bool enable_graph_optimization = [cuda_ctx] {
        const char * env = getenv("GGML_CUDA_GRAPH_OPT");
        if (env != nullptr) {
            return atoi(env) == 1;
        }
        const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
        return GGML_CUDA_CC_IS_RDNA3_5(cc);
    }();
"""


PATCH = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="Default GGML_CUDA_GRAPH_OPT to enabled on RDNA3.5, env "
                "var stays an override everywhere (amd-ecosystem PR #56 / "
                "RD44)",
    edits=(
        Edit(
            id="rd44-graph-opt-default-rdna35",
            anchor=re.escape(_ENABLE_HEAD) + r"[ ]{15,40}" + re.escape(_ENABLE_TAIL),
            rationale="enable_graph_optimization: default on for gfx1151 "
                      "when the env var is unset, keep it an explicit "
                      "override everywhere",
            mode="replace",
            text=_ENABLE_NEW,
            guard=r"return GGML_CUDA_CC_IS_RDNA3_5\(cc\);",
        ),
    ),
)

PATCHES = [PATCH]
