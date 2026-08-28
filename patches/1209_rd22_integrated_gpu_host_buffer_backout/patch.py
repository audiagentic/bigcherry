"""RD22: back out integrated-GPU host buffers on HIP (fork divergence from PR #24233).

Provenance (group 'rdna-boosts' patches are external backports; the
machine-readable PROVENANCE dict below is cross-checked against
external-sources.toml by tools/tests/test_external_sources.py):

  source:            stew675-rdna-boosts
  repo:              https://github.com/stew675/llama.cpp
  locator:           rdna-boosts (branch name is a locator only, NOT identity)
  fork commit:       507f2e267 (snapshot v2; no v1 counterpart -- the v1
                     ledger's item 8, 85a9069a0, is the pre-rebase identity
                     of the SAME logical change, content-identical per
                     git patch-id)
                     "cuda : back out integrated-GPU host buffers on HIP
                     (PR #24233)"
  reviewed snapshot: v2 -- head 9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22
                     on base 4df29be4f4c3673f428170fda944a5b19f743bb8
  plan item:         RD22 (docs/planning/active/rdna-boost-experiments/RD22.md)
  mainline status:   NOT merged into ggml-org/llama.cpp master as of tip
                     6d0549831 (git cherry patch-id check, 2026-08-18).
                     NOTE the direction: mainline PR #24233 RESTORED
                     prop.integrated on HIP; this fork commit is a
                     deliberate divergence BACK AWAY from mainline. A pin
                     bump therefore does NOT absorb it -- it would
                     RE-ENABLE the broken path. Re-check before every
                     pin bump.

What it does (hardware-scoped correctness, not performance):
  Upstream PR #24233 restored `prop.integrated` on HIP builds, enabling
  the CUDA host-buffer (zero-copy UMA weights) path on APUs. On the
  fork author's Strix Halo iGPU that path corrupts full-model results
  under async execution (PPL 5.9243 -> 8.51+ without
  HIP_LAUNCH_BLOCKING). Forcing integrated=false restores async-safe
  operation in every load mode at no decode/prefill cost.

  The change is a no-op on discrete GPUs: on those, the host-buffer
  path is not selected anyway, and the value is only consumed at the
  buft/host-buffer decision sites (the `integrated` flag used in the
  copy-scheduling region and the is_uma check).

Porting notes:
  - Ported VERBATIM from the fork commit: one line replaced inside
    ggml_cuda_init()'s #if defined(GGML_USE_HIP) branch, plus the fork's
    own explanatory comment.
  - The anchor region (#if/#else/#endif around line 330 of
    ggml-cuda.cu) is byte-identical on the pinned llama.cpp
    4801e3c567d5131dd41b387df5f2d4b1370d92be and survives the
    framework patches (no framework patch touches this region).
  - The #else branch line and the patched #if branch line become
  TEXTUALLY IDENTICAL after the patch (both say `integrated = false;`
    with the same comment). That is intentional -- it mirrors the fork
    exactly -- and is why the already-applied guard keys on the
    fork's unique comment line, not on the assignment line.

Hardware status (RD22's own acceptance):
  The corruption was observed on a Strix Halo iGPU (gfx1151 RDNA3.5).
  Brutus has no discrete evidence path for it yet; device 2 (gfx1201,
  integrated on that board) is the candidate reproduction target.
  Per the item's standard this patch is evidence-only until an
  integrated-GPU PPL/output-integrity run demonstrates the behavior on
  OUR hardware -- it must never be promoted as a global workaround.

Isolation and promotion (first-sweep policy, RD review 2026-08-18):
  - GROUP 'rdna-boosts' + STATE 'untested' keeps this OUT of the
    production 'framework' and 'validated-enhancements' patch-sets.
  - On promotion (only after iGPU evidence): set STATE = 'validated'
    and add this patch id to [patch-set.validated-enhancements] in
    recipes.toml. Discrete-GPU builds are expected to show zero
    difference (that no-regression is part of the bench).

Maintenance (future pin bumps / fork movement):
  - On every pin bump, verify upstream has not changed the
    #if defined(GGML_USE_HIP) integrated line (mainline PR #24233
    already set it to prop.integrated; a future mainline commit could
    restructure the block). Re-derive from the tracked fork commit in
    external-sources.toml; run `python -m bigcherry sources check`.
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

PROVENANCE = {
    "source-id": "stew675-rdna-boosts",
    "plan-item": "RD22",
    "fork-commit": "507f2e26719b33a7c82429609bc7806aa60f27e6",
    "fork-commit-title": "cuda : back out integrated-GPU host buffers on HIP (PR #24233)",
    "original-commit": "85a9069a0a01a9949ee0df8bc0668c2ee0e2af12",
    "snapshot-head": "9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22",
    "snapshot-base": "4df29be4f4c3673f428170fda944a5b19f743bb8",
    "adaptations": [],
}


_OLD = """#if defined(GGML_USE_HIP)
        info.devices[id].integrated = prop.integrated;
#else"""

_NEW = """#if defined(GGML_USE_HIP)
        // Fork divergence from PR #24233: integrated=true enables the CUDA
        // host-buffer path (zero-copy UMA weights) on APUs, which corrupts
        // full-model results under async execution on this box (PPL 5.9243
        // -> 8.51+ without HIP_LAUNCH_BLOCKING). The back-out restores
        // async-safe operation at no decode/prefill cost.
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
#else"""


PATCH = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="Back out integrated-GPU host buffers on HIP: force "
                "devices[id].integrated = false in ggml_cuda_init "
                "(rdna-boosts 507f2e267 / RD22)",
    edits=(
        Edit(
            id="rd22-integrated-backout",
            anchor=re.escape(_OLD),
            rationale="ggml_cuda_init: the HIP branch that PR #24233 "
                      "restored to prop.integrated; the fork back-outs the "
                      "host-buffer path there",
            mode="replace",
            text=_NEW,
            # The assignment line alone is ambiguous post-patch (the #else
            # branch already carries the identical line), so the
            # already-applied guard keys on the fork's unique comment.
            guard=r"Fork divergence from PR #24233",
        ),
    ),
)

PATCHES = [PATCH]
