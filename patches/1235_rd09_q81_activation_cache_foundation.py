"""RD09 stage 1: per-graph Q8_1 activation-quantization cache -- foundation
only. No caller in mmvq.cu references this cache yet (that wiring is RD09
stage 2, a separate future patch); this patch only adds the cache's own
implementation files to the HIP build. Until a caller exists, the added
code is unreachable dead weight, so this patch cannot change any model
output or timing by construction -- the isolated bench for THIS patch is
therefore expected to show zero difference from bigcherry-native.

Provenance (group 'rdna-boosts' patches are external backports; the
machine-readable PROVENANCE dict below is cross-checked against
external-sources.toml by tools/tests/test_external_sources.py):

  source:            stew675-rdna-boosts
  repo:              https://github.com/stew675/llama.cpp
  locator:           rdna-boosts (branch name is a locator only, NOT identity)
  fork commit:       299f6eaf73b5eeb888bd94eaa66122d003136e6a
                     (rebased v2 snapshot of "CUDA: cache quantized Q8_1
                     matmul inputs per graph")
  original commit:   ff6fde5046ffb86672e05da640d2bfb20d4bfdfc
                     (pre-rebase; the plan item's own description cited only
                     a truncated "ff6fde5", resolved to this full hash and
                     verified content-identical to the v2 rebase via the v2
                     snapshot's own patch-id audit, 2026-08-24)
  reviewed snapshot: v2 -- head 9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22
                     on base 4df29be4f4c3673f428170fda944a5b19f743bb8
  plan item:         RD09 (docs/planning/active/rdna-boost-experiments/RD09.md)
  mainline status:   NOT merged into ggml-org/llama.cpp master as of tip
                     6d0549831 (same base check as RD19, 2026-08-18)

What it does:
  The fork's own commit is small (89 additions / 7 deletions across
  common.cuh, ggml-cuda.cu, mmvq.cu) and NOT a verbatim port here: this
  patch reimplements the same mechanism -- reuse one Q8_1 quantization of
  an activation tensor across every MMVQ consumer that needs the identical
  quantized data within the same graph evaluation, instead of
  re-quantizing per node -- with two required adaptations found during
  design review (dev-gpt-agent, session ses_76b0fef0c94c434a,
  req_60a41664e0de43d6) that the fork's own implementation does not have:

    1. Cache key includes the exact view data address/offset, not just the
       view-root pointer plus shape/stride/stream. The fork's key omits the
       offset, so two views of the SAME root tensor with identical
       shape/stride but different byte offsets can collide there -- a real
       false-hit bug in the source, not a hypothetical, and precisely the
       failure mode this plan item's own risk statement ("stale cache
       entries could corrupt outputs") warns against.
    2. Backing storage is a bounded set of stable, NEVER-relocated slabs
       (append a new slab on growth; existing slabs are never copied,
       freed, or moved), not the fork's single arena that grows by
       allocate-copy-free. HIP/CUDA graph capture bakes pointer values into
       the captured graph; a relocating arena would silently corrupt every
       subsequent replay once it needed to grow.

  This stage adds only the cache's own implementation
  (src/ggml/src/ggml-cuda/hip-q81-cache.{h,cpp}: generation-scoped
  find/reserve/publish API, stable slab allocator, stats counters, the
  GGML_HIP_Q8_1_CACHE_MODE=off|on|verify env gate) and wires the two new
  files into the HIP backend's CMake build. It does not touch mmvq.cu, the
  0200 dispatch hook, or any other live code path -- see "What it does
  NOT do yet" below.

  Fork-reported effect (gfx1201, unspecified model, decode-heavy workload):
  not directly comparable here since this stage adds no caller; RD09's own
  causal bench (stage 5 of the staged plan below) will measure real
  quantize-launch reduction once stage 2 wires the cache into
  ggml_cuda_mul_mat_vec_q().

What it does NOT do yet (see docs/planning/active/rdna-boost-experiments/RD09.md
for the full 7-stage plan and its exit gates):
  - Stage 2 (MMVQ integration): replace the current local Q8_1
    materialization block in ggml_cuda_mul_mat_vec_q() (mmvq.cu, the single
    existing call site of quantize_row_q8_1_cuda) with cache
    find/reserve/publish calls. This is the change that would actually
    alter model behavior if it had a bug -- deliberately isolated into its
    own future patch so this foundation stage can be reviewed and tested
    with zero behavioral risk.
  - Stage 3 (adversarial correctness): the verify-mode byte-compare
    harness and the full same-root/different-offset / cross-stream /
    generation-boundary / capacity-exhaustion correctness matrix from the
    plan item.
  - Stage 4 (capture/topology): HIP graph warm-up/capture/replay
    validation, dual-XTX tensor-split isolation.
  - Stage 5 (causal bench): same-binary cache-off-vs-on A/B, quantize
    launch counts, TG/PP/memory.

Porting notes:
  - No textual anchor into any vendor llama.cpp SOURCE file. The only
    vendor file touched is ggml/src/ggml-hip/CMakeLists.txt, to add the two
    new source files to GGML_SOURCES_ROCM -- an addition, not a
    modification of any existing line, so it cannot conflict with any
    other patch's edits to that file.
  - The new files always compile in (no build-time flag): the
    GGML_HIP_Q8_1_CACHE_MODE env var is a pure runtime toggle, defaulting
    to off, and off must be byte-for-byte identical to the pre-cache
    behavior. This is deliberate -- it is what makes the eventual stage 5
    bench a true same-binary A/B (flip an env var, not rebuild) rather than
    a cross-build comparison.

Isolation and promotion (first-sweep policy, matches RD19's own convention):
  - GROUP 'rdna-boosts' + STATE 'untested' keeps this OUT of the production
    'framework' and 'validated-enhancements' patch-sets (both use exact
    patch-id lists in recipes.toml). It is built ONLY when explicitly
    selected:
        python -m bigcherry apply --groups core,rdna-boosts \
                                  --states validated,untested
  - Isolated test = bigcherry-native (framework only) bench vs this build
    (framework + this patch). Since stage 1 adds no caller, the isolated
    bench for THIS patch alone is a build-succeeds/host-tests-pass check,
    not a performance comparison -- performance evidence arrives with
    stage 2+.
  - On eventual full-foundation promotion (after stage 6's exit gate):
    set STATE = 'validated' and add this patch id to
    [patch-set.validated-enhancements] in recipes.toml.

Maintenance (future pin bumps / fork movement):
  - Run `python -m bigcherry sources check` to detect: (a) the fork branch
    moving or being rebased again, (b) this mechanism landing in mainline
    (patch-id match against ff6fde5046ff... or 299f6eaf...), which would
    make this patch redundant on the next pin bump, (c) content drift of
    the fork commit vs the tracked snapshot.
"""

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

PROVENANCE = {
    "source-id": "stew675-rdna-boosts",
    "plan-item": "RD09",
    "fork-commit": "299f6eaf73b5eeb888bd94eaa66122d003136e6a",
    "fork-commit-title": "CUDA: cache quantized Q8_1 matmul inputs per graph",
    "original-commit": "ff6fde5046ffb86672e05da640d2bfb20d4bfdfc",
    "snapshot-head": "9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22",
    "snapshot-base": "4df29be4f4c3673f428170fda944a5b19f743bb8",
    "adaptations": [
        "cache key strengthened with the exact view data address/offset "
        "(the fork's key omits it, allowing a same-root/different-offset "
        "false-hit collision)",
        "backing storage reimplemented as bounded, never-relocated stable "
        "slabs instead of the fork's relocatable grow-by-copy arena, for "
        "HIP/CUDA graph-capture pointer stability",
        "stage 1 (this patch) adds only the cache implementation with no "
        "caller; the fork's commit wires straight into mmvq.cu in one "
        "step, deliberately split here into a separate future stage 2 "
        "patch so the zero-behavioral-risk foundation can be reviewed and "
        "tested independently of the actual MMVQ integration",
    ],
}

_CMAKE_SOURCES_ANCHOR = (
    r'file\(GLOB   SRCS "\.\./ggml-cuda/template-instances/mmf\*\.cu"\)\n'
    r'list\(APPEND GGML_SOURCES_ROCM \$\{SRCS\}\)'
)

_CMAKE_SOURCES_NEW = """
    # bigcherry (RD09 stage 1): per-graph Q8_1 activation-quantization
    # cache foundation. Always compiled in -- GGML_HIP_Q8_1_CACHE_MODE is a
    # pure runtime toggle (default off, byte-for-byte identical to the
    # pre-cache path), not a build-time flag, so a stage 2+ bench can be a
    # same-binary A/B. No caller references this yet; see RD09.md.
    list(APPEND GGML_SOURCES_ROCM
        "../ggml-cuda/hip-q81-cache.cpp")"""

CMAKE_PATCH = FilePatch(
    path="ggml/src/ggml-hip/CMakeLists.txt",
    description="RD09 stage 1: add the Q8_1 activation cache foundation to the HIP source list",
    edits=(
        Edit(
            id="rd09-q81-cache-source",
            anchor=_CMAKE_SOURCES_ANCHOR,
            rationale="the end of the GGML_SOURCES_ROCM glob block -- the same stable "
                      "anchor 0100_cmake_options.py's own dispatch-sources edit uses, "
                      "so this addition composes with it regardless of application order",
            text=_CMAKE_SOURCES_NEW,
            guard=r"hip-q81-cache\.cpp",
        ),
    ),
)

PATCHES = [CMAKE_PATCH]
