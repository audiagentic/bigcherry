"""RD19: skip the Meta device wrapper when tensor-splitting a single GPU.

Provenance (group 'rdna-boosts' patches are external backports; the
machine-readable PROVENANCE dict below is cross-checked against
external-sources.toml by tools/tests/test_external_sources.py):

  source:            stew675-rdna-boosts
  repo:              https://github.com/stew675/llama.cpp
  locator:           rdna-boosts (branch name is a locator only, NOT identity)
  fork commit:       3c48ecd635022522b41a622aae28b565497abef5
                     "llama : skip the Meta device wrapper with a single GPU"
  original commit:   a6e774d41b28c2f9f1a36e3752064ed2a26d3e4d
                     (pre-rebase snapshot v1; content-identical to the fork
                     commit above, verified by git patch-id on 2026-08-18)
  reviewed snapshot: v2 -- head 9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22
                     on base 4df29be4f4c3673f428170fda944a5b19f743bb8
  plan item:         RD19 (docs/planning/active/rdna-boost-experiments/RD19.md)
  mainline status:   NOT merged into ggml-org/llama.cpp master as of tip
                     6d0549831 (git cherry patch-id check, 2026-08-18)

What it does:
  -s tensor creates a Meta device that wraps the GPU backends and splits the
  graph into subgraphs at every partial-split node -- even when only ONE
  device is available and no splitting is possible. Each subgraph is a
  separate graph compute call on the CUDA/HIP backend, multiplying the
  per-token launch overhead and (per the fork's note) clearing the Q8_1
  quantize cache between subgraphs.

  This patch uses the plain device when n_devices == 1 in both branches of
  llama_prepare_model_devices (explicit device list, and default device
  selection); the Meta path is untouched for multi-GPU tensor parallelism.

  Fork-reported effect (gfx1201, Qwen3.6-27B-Q6_K, BF16 KV, tg64):
  d8192 23.48 -> 23.91 t/s (+1.8%), d65536 20.57 -> 20.89 (+1.6%),
  d256 24.14 -> 24.49 (+1.4%). Two-GPU meta path re-verified working.
  This is the fork author's number, not ours: BigCherry must reproduce it
  in the isolated bench before any promotion claim.

Porting notes:
  - Ported VERBATIM from the fork commit. Both hunks anchor byte-identical on
    the pinned llama.cpp 4801e3c567d5131dd41b387df5f2d4b1370d92be
    (llama_prepare_model_devices, ~line 162 and ~line 206); zero adaptation.
    src/llama.cpp drifted +16/-5 between our pin and the fork's base, but
    not in these two regions.
  - No dependency on any other BigCherry patch. The two hunks do not touch
    anything the framework dispatch patches (0200/0700/0900) anchor.

Isolation and promotion (first-sweep policy, RD review 2026-08-18):
  - GROUP 'rdna-boosts' + STATE 'untested' keeps this OUT of the production
    'framework' and 'validated-enhancements' patch-sets (both use exact
    patch-id lists in recipes.toml). It is built ONLY when explicitly
    selected:
        python -m bigcherry apply --groups core,rdna-boosts \
                                  --states validated,untested
  - Isolated test = bigcherry-native (framework only) bench vs this build
    (framework + this patch). No other RD patch is composed in at this
    stage; conflicts with other patches are only assessed IF this one proves
    worth promoting.
  - On promotion: set STATE = 'validated' and add this patch id to
    [patch-set.validated-enhancements] in recipes.toml (that set is the
    designated home for promoted enhancement patches).

Maintenance (future pin bumps / fork movement):
  - If llama_prepare_model_devices changes shape upstream, re-derive the two
    hunks from the tracked fork commit (or its successor) in
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
    "plan-item": "RD19",
    "fork-commit": "3c48ecd635022522b41a622aae28b565497abef5",
    "fork-commit-title": "llama : skip the Meta device wrapper with a single GPU",
    "original-commit": "a6e774d41b28c2f9f1a36e3752064ed2a26d3e4d",
    "snapshot-head": "9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22",
    "snapshot-base": "4df29be4f4c3673f428170fda944a5b19f743bb8",
    "adaptations": [],
}


# Anchor matched against the NOISE-STRIPPED copy of the file (patcher.py):
# comments and string literals are blanked to spaces there, so the two
# LLAMA_LOG_INFO message strings are matched with .* rather than their
# literal text. The replacement below (mode=replace) is inserted VERBATIM at
# the same offsets in the real text, so it keeps the real message strings.
_HUNK1_ANCHOR = r"""            LLAMA_LOG_INFO\(.*__func__, n_devs\);
            for \(size_t i = 0; i < n_devs; \+\+i\) \{
                LLAMA_LOG_INFO\(.*__func__, i, ggml_backend_dev_name\(params\.devices\[i\]\)\);
            \}
            model->get_split_state_ud\.n_devices = n_devs;
            model->get_split_state_ud\.model = model;
            model->devices\.push_back\(\{
                true, ggml_backend_meta_device\(
                params\.devices, n_devs, llama_meta_device_get_split_state, &model->get_split_state_ud\)
            \}\);"""

_HUNK1_NEW = """            if (n_devs == 1) {
                // single device: no tensor splitting is possible, skip the Meta wrapper
                model->devices.push_back({false, params.devices[0]});
            } else {
                LLAMA_LOG_INFO("%s: creating a Meta device with %zu devices\\n", __func__, n_devs);
                for (size_t i = 0; i < n_devs; ++i) {
                    LLAMA_LOG_INFO("%s: - device %zu: %s\\n", __func__, i, ggml_backend_dev_name(params.devices[i]));
                }
                model->get_split_state_ud.n_devices = n_devs;
                model->get_split_state_ud.model = model;
                model->devices.push_back({
                    true, ggml_backend_meta_device(
                    params.devices, n_devs, llama_meta_device_get_split_state, &model->get_split_state_ud)
                });
            }
"""

_HUNK2_OLD = """            GGML_ASSERT(!devs.empty());
            model->get_split_state_ud.n_devices = devs.size();
            model->get_split_state_ud.model     = model;
            gpus.push_back({
                true, ggml_backend_meta_device(
                devs.data(), devs.size(), llama_meta_device_get_split_state, &model->get_split_state_ud)
            });
"""

_HUNK2_NEW = """            GGML_ASSERT(!devs.empty());
            if (devs.size() == 1) {
                // single device: no tensor splitting is possible, skip the Meta wrapper
                gpus.push_back({false, devs[0]});
            } else {
                model->get_split_state_ud.n_devices = devs.size();
                model->get_split_state_ud.model     = model;
                gpus.push_back({
                    true, ggml_backend_meta_device(
                    devs.data(), devs.size(), llama_meta_device_get_split_state, &model->get_split_state_ud)
                });
            }
"""


PATCH = FilePatch(
    path="src/llama.cpp",
    description="Skip the Meta device wrapper when tensor-splitting a single "
                "GPU (rdna-boosts 3c48ecd63 / RD19)",
    edits=(
        Edit(
            id="rd19-single-device-explicit-list",
            anchor=_HUNK1_ANCHOR,
            rationale="llama_prepare_model_devices: the explicit-device "
                      "tensor-split branch (LLAMA_SPLIT_MODE_TENSOR with "
                      "params.devices set), where the Meta device is created",
            mode="replace",
            text=_HUNK1_NEW,
            # New-shape ONLY: the guard detects 'already applied'. It must not
            # match the unpatched block (verified: 0 matches in the pinned
            # file) or a fresh apply would be skipped as already-applied.
            guard=r"model->devices\.push_back\(\{false, params\.devices\[0\]\}\);",  # noqa: E501
        ),
        Edit(
            id="rd19-single-device-default-selection",
            anchor=re.escape(_HUNK2_OLD),
            rationale="llama_prepare_model_devices: the default device "
                      "selection branch (all backends), second Meta device "
                      "creation site",
            mode="replace",
            text=_HUNK2_NEW,
            # New-shape ONLY (see hunk 1): 0 matches in the pinned file.
            guard=r"gpus\.push_back\(\{false, devs\[0\]\}\);",  # noqa: E501
        ),
    ),
)

PATCHES = [PATCH]
