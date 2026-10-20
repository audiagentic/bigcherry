# Radiance on BigCherry: runbook

How the radiance engine is built, configured and measured through the same platform as llama.cpp (RAD01; design in
[PLATFORM_DESIGN.md](PLATFORM_DESIGN.md)). The measurement log is
[engines/radiance/README.md](../../../../engines/radiance/README.md).

## What lives where

| Piece | Location |
|---|---|
| Engine declaration (upstream, layout, build, serve, components) | `engines/radiance/engine.toml` |
| Recipes: pin, patch sets, sources, builds, experiments, runtime profiles, platform | `engines/radiance/recipes.toml` |
| Engine patches (packages against the pinned radiance) | `engines/radiance/patches/<id>/` |
| libr3, the gfx1100 kernel plugin (component, not a patch) | `engines/radiance/components/libr3/` |
| libr3 runtime settings (R3_*: default, status, activation marker) | `engines/radiance/components/libr3/component.toml` |
| moe-remote (the remote-MoE helper `r3_moe_worker` and the `libr3moe` plugin; R3_MOE_* settings) | `engines/radiance/components/moe-remote/` and its `component.toml` |
| A/B configurations | `tools/lab/radiance/*.json` |

## One-time host setup (build host)

1. Upstream mirror, beside llama.cpp's under the work root:

       git clone --bare <a radiance clone> <work>/upstream/radiance.git
       git -C <work>/upstream/radiance.git remote set-url origin https://codeberg.org/StillDeadcode/radiance.git

   The pin (`engines/radiance/recipes.toml` `pinned`) must resolve in it.
2. Host compilers in `config/environment.local.toml` `[env]` (radiance needs GCC 14 for its host code):

       RADIANCE_HOST_CC = "<gcc-14>"
       RADIANCE_HOST_CXX = "<g++-14>"

3. ROCm on `PATH` when building (`export ROCM_PATH=/opt/rocm-7.2.4 PATH=$ROCM_PATH/bin:$PATH`); radiance's own CMake
   finds HIP from it. A worktree build points `BIGCHERRY_ENVIRONMENT` at the host file.

## Build

    python -m bigcherry build --lane radiance-lab:stock:linux-radiance

The lane materialises the pin with the source's patch sets applied by the patch system, builds `bin/radiance` and
radiance's default target (its plugins, architectures and quantisers into `radiance_home`), then builds each
component (libr3) against the engine installed from that build and the materialised source, and copies its outputs
into `radiance_home`. The build's identity includes the runtime targets and the components' content.

Sources: `radiance-native` (the pin), `radiance` (validated patch sets), `radiance-lab` (evaluated engine fixes).
One engine patch per experiment (`[experiment.*]`).

## Measure

A/B through `bigcherry ab-benchmark --server-config <json> --pairs N`. The configuration names `"engine": "radiance"`;
arms run in mode `stock` with `execution_evidence: "observe"` (device attestation is parsed for llama.cpp only).
Quality gates: a `quality` block in the same configuration (`tools/lab/radiance/helper-kld.json`) runs radiance's KL
mode per arm, `tools/bigcherry/experiment/engine_quality.py` parses the `--kld-out` JSON and gates a subject against
its control; a component setting with an `activation` marker yields activation evidence from the
two arms' server logs.

## Profiles

`[runtime-profile.*]` in the recipes hold only what they switch: `server-args` and an `environment` checked against
the components (no undeclared R3_*, no restated default). `flashnext-2xtx` and `qwen38-27b-2xtx` are the starting
profiles.
