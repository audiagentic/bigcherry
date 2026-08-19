"""RD21: gfx1151 (RDNA3_5) MMVQ table with nwarps=2 for Q8_0 decode.

Provenance (group 'rdna-boosts' patches are external backports; the
machine-readable PROVENANCE dict below is cross-checked against
external-sources.toml by tools/tests/test_external_sources.py):

  source:            stew675-rdna-boosts
  repo:              https://github.com/stew675/llama.cpp
  locator:           rdna-boosts (branch name is a locator only, NOT identity)
  fork commit:       1818c3b3704a623e33d73b6cb0b633400ef70b14
                     (snapshot v2; v1 ledger item 9, 9f807c21d, is the
                     pre-rebase identity of the SAME logical change,
                     content-identical per git patch-id)
                     "cuda : gfx1151 mmvq table with nwarps=2 for Q8_0
                     decode"
  reviewed snapshot: v2 -- head 9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22
                     on base 4df29be4f4c3673f428170fda944a5b19f743bb8
  plan item:         RD21 (docs/planning/active/rdna-boost-experiments/RD21.md)
  mainline status:   NOT merged into ggml-org/llama.cpp master as of tip
                     6d0549831 (git cherry patch-id check, 2026-08-18)

WHAT IS ACTUALLY PORTED (RD25 bake-in -- deviates from the single-commit
diff by design, per the RD25 rule recorded 2026-08-19):
  The nwarps table block is taken from the BRANCH-TIP (9e46e1fd) state,
  i.e. fork commit 8cdf1ab08's fix is baked in:
      if (ncols_dst <= MMVQ_MAX_BATCH_SIZE)   <- tip state (was ncols_dst == 1)
  because 8cdf1ab08 extends the gfx1151 table to the whole mmvq range so
  the speculative-verify batch accumulates bit-identically to decode.
  Porting the pre-fix table would reintroduce the inconsistency the fix
  exists to remove; the pre-fix state should never exist in the tree.
  The other three hunks (enum entry, device table, host table) are
  untouched by 8cdf1ab08 and are ported verbatim from 1818c3b37.

What it does (hardware-scoped performance):
  gfx1151 (Strix Halo iGPU, RDNA3_5) previously fell into the RDNA2
  mmvq table. A dedicated table gives nwarps=2 for Q8_0 decode matmuls
  (swept 2025-08 on the fork: ~+0.6% decode on Qwen3.6-35B-A3B Q8_0;
  nwarps=4 regresses). On every other architecture the table resolves
  exactly as before (the RDNA2/RDNA3_5 combined branch is split into
  two branches with identical behavior for RDNA2).

Porting notes:
  - All four anchor regions are byte-identical on the pinned llama.cpp
    4801e3c567d5131dd41b387df5f2d4b1370d92be (mmvq.cu) and survive the
    framework patches: 0600/0650 edit mmvq.cu's launch/geometry region,
    not the table/enum region.
  - MMVQ_MAX_BATCH_SIZE (used by the baked-in fix) is a pre-existing
    macro in mmvq.cuh -- no new symbols.
  - No BigCherry patch and no other RD item anchors in these four
    regions (the mmvq-family RD items RD08/RD12/RD27 anchor elsewhere in
    the file; experimental-vs-experimental collisions are deferred by
    first-sweep policy).

Hardware status (RD21's own acceptance):
  DEFERRED on hardware: requires gfx1151. Brutus has gfx1100/gfx1201/
  gfx1030. The patch is behavior-neutral on all three (the RDNA3_5
  branch cannot resolve), so the bench reduces to: no-regression on
  gfx1100 (primary) + explicit non-selection proof on the Brutus
  devices; the performance claim stays unvalidated until gfx1151
  hardware exists. Per the item's standard: no unsupported extrapolation.

Isolation and promotion (first-sweep policy, RD review 2026-08-18):
  - GROUP 'rdna-boosts' + STATE 'untested' keeps this OUT of the
    production 'framework' and 'validated-enhancements' patch-sets.
  - Promotion is gated on gfx1151 evidence (RD21 stays open as a
    hardware-deferred item even after the patch lands).

Maintenance (future pin bumps / fork movement):
  - calc_nwarps and the two get_device_table_id overloads are touched by
    upstream table updates; re-derive from the tracked fork commits
    (1818c3b37 + 8cdf1ab08, both in external-sources.toml) and run
    `python -m bigcherry sources check` before every pin bump.
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

PROVENANCE = {
    "source-id": "stew675-rdna-boosts",
    "plan-item": "RD21",
    "fork-commit": "1818c3b3704a623e33d73b6cb0b633400ef70b14",
    "fork-commit-title": "cuda : gfx1151 mmvq table with nwarps=2 for Q8_0 decode",
    "original-commit": "9f807c21d527e061febe13a2cb29dc97bdab3f4a",
    "snapshot-head": "9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22",
    "snapshot-base": "4df29be4f4c3673f428170fda944a5b19f743bb8",
    "adaptations": [
        "calc_nwarps RDNA3_5 table block taken from branch-tip (9e46e1fd) "
        "state: bakes in 8cdf1ab08's ncols_dst <= MMVQ_MAX_BATCH_SIZE "
        "extension (RD25 rule) so the pre-fix ncols_dst == 1 state never "
        "exists in the tree",
    ],
}


# --- hunk 1: enum entry ------------------------------------------------
_ENUM_OLD = """    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4
"""

_ENUM_NEW = """    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA3_5,
    MMVQ_PARAMETERS_RDNA4
"""

# --- hunk 2: device-side table resolution ------------------------------
_DEV_OLD = """#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
"""

_DEV_NEW = """#elif defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA3_5;
#elif defined(RDNA2)
    return MMVQ_PARAMETERS_RDNA2;
"""

# --- hunk 3: host-side table resolution --------------------------------
_HOST_OLD = """    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
"""

_HOST_NEW = """    if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA3_5;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
"""

# --- hunk 4: the table itself (TIP STATE, RD25 baked in) ---------------
# Inserted between the RDNA3_0 table and the TURING table in calc_nwarps.
_NWARP_OLD = """        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {"""

_NWARP_NEW = """        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_5) {
        // gfx1151 (Strix Halo iGPU): nwarps=1 (the RDNA2 table) underutilizes the
        // wave32 datapath on the large-K decode matmuls; nwarps=8 (the RDNA3_0
        // table) over-parallelizes the small ones. Swept 2025-08: nwarps=2 wins
        // (~+0.6% decode on Qwen3.6-35B-A3B Q8_0), nwarps=4 regresses.
        // Apply to the whole mmvq range (ncols_dst 1..8), not just decode: the
        // speculative verify batch (n_draft+1 tokens) must use the same nwarps
        // as decode so its per-row dot-product accumulation is bit-identical.
        if (ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
            switch (type) {
                case GGML_TYPE_Q8_0:
                    return 2;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {"""


PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmvq.cu",
    description="Add the gfx1151 (RDNA3_5) mmvq parameter table with "
                "nwarps=2 for Q8_0 decode (rdna-boosts 1818c3b37 / RD21, "
                "RD25 tip-state baked in)",
    edits=(
        Edit(
            id="rd21-enum-entry",
            anchor=re.escape(_ENUM_OLD),
            rationale="mmvq_parameter_table_id enum: add the RDNA3_5 "
                      "table id between RDNA3_0 and RDNA4 (fork order)",
            mode="replace",
            text=_ENUM_NEW,
            guard=r"MMVQ_PARAMETERS_RDNA3_5,",
        ),
        Edit(
            id="rd21-device-table",
            anchor=re.escape(_DEV_OLD),
            rationale="get_device_table_id (device-side): split the "
                      "RDNA2||RDNA3_5 branch so RDNA3_5 resolves to its "
                      "own table",
            mode="replace",
            text=_DEV_NEW,
            guard=r"#elif defined\(RDNA3_5\)\n    return MMVQ_PARAMETERS_RDNA3_5;",
        ),
        Edit(
            id="rd21-host-table",
            anchor=re.escape(_HOST_OLD),
            rationale="get_device_table_id(int cc) (host-side): split the "
                      "RDNA2||RDNA3_5 branch so RDNA3_5 resolves to its "
                      "own table",
            mode="replace",
            text=_HOST_NEW,
            guard=r"if \(GGML_CUDA_CC_IS_RDNA3_5\(cc\)\) \{",
        ),
        Edit(
            id="rd21-nwarps-table",
            anchor=re.escape(_NWARP_OLD),
            rationale="calc_nwarps: insert the RDNA3_5 table between the "
                      "RDNA3_0 table and the TURING table (fork position); "
                      "tip-state body per the RD25 bake-in rule",
            mode="replace",
            text=_NWARP_NEW,
            guard=r"if \(table_id == MMVQ_PARAMETERS_RDNA3_5\)",
        ),
    ),
)

PATCHES = [PATCH]
