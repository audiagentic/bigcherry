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
  - DEVIATES FROM THE FORK by design (bug found in GPT-auto-agent review,
    2026-08-20): the fork's own PR #56 caches the RDNA3.5 architecture
    check inside a function-local `static` initializer that captures
    `cuda_ctx` by value. C++ function-local `static` semantics run the
    initializer exactly ONCE per process, on the first call -- capturing
    `cuda_ctx` does not make it re-evaluate per device. On a process that
    sees more than one GPU architecture (this project's own Brutus fleet:
    gfx1100/gfx1201/gfx1030 simultaneously), the first device to call this
    function would silently fix the default for every other device too.
    `ggml_backend_cuda_graph_optimize` immediately returns a few lines
    below if `ggml_backend_cuda_get_device_count() != 1`, so this bug is
    UNREACHABLE today in this exact call path (the function only ever
    proceeds past the check in a single-visible-device process, where
    `cuda_ctx->device` cannot vary between calls) -- but it is still wrong
    as written and would silently become live if that guard is ever
    loosened by a future patch. Fixed here rather than ported faithfully:
    only the env-var override is cached process-wide (correct -- the
    environment does not change mid-process); the architecture check
    itself runs on every call, uncached.
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

# RE40 (external patch-management review, 2026-08-20): "depends on RD39-RD43
# (patches 1215+1216) already being applied" was documented prose; REQUIRES
# makes it real and enforced (patchset.resolve_exact(), already called by
# resolve_lane's experiment= path). Backfilled safely: config/recipes.toml's
# only entry naming 1217 already includes both 1215 and 1216.
REQUIRES = ("1215_rd394041_amd_stream_moe_overlap", "1216_rd43_concurrent_join_fusion_guard")

PROVENANCE = {
    "source-id": "amd-ecosystem-llama-cpp",
    "plan-item": "RD44",
    "fork-commit": "6e9f948a0d8b99db2bce0a83fc021efc2064ef9b",
    "fork-commit-title": "ggml-cuda: enable GGML_CUDA_GRAPH_OPT by default on RDNA3.5",
    "snapshot-head": "58ab0a5f2ce3f426d657d55647846b03fbc1a20b",
    "snapshot-base": "58ab0a5f2ce3f426d657d55647846b03fbc1a20b",
    "adaptations": [
        "Fixed a real multi-device bug in the fork's own implementation: "
        "its function-local static caches the architecture check "
        "process-wide from the first caller, not per-device. Only the "
        "env-var override is cached here; the architecture check runs on "
        "every call. See the porting notes above for the full analysis "
        "and why the bug is currently unreachable but was fixed anyway.",
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

_ENABLE_NEW = """    // Only the env-var override is safe to cache process-wide (the
    // environment does not change mid-process). The architecture check
    // must run on every call, uncached -- a function-local `static`
    // capturing cuda_ctx does NOT re-evaluate per device; it runs its
    // initializer once, on the first caller, for the whole process. On a
    // process that sees more than one GPU architecture the first device
    // to call this would silently fix the default for every other device
    // too (see the patch module's porting notes for the full analysis;
    // this deviates from the fork's own PR #56, which has this bug).
    static const int env_graph_opt_override = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr ? (atoi(env) == 1 ? 1 : 0) : -1;
    }();
    const bool enable_graph_optimization = env_graph_opt_override >= 0
        ? (env_graph_opt_override != 0)
        : GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc);
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
            guard=r"env_graph_opt_override >= 0",
        ),
    ),
)

PATCHES = [PATCH]
