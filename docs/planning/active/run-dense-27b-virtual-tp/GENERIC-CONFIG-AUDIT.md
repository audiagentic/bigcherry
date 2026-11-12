# Generic-configuration audit (2026-11-12)

The rule: the engine and tooling are generic. Models, GPU layouts and topologies, tensor-parallel splits, helper
configurations, expert placement and swapping, kernel choices and tuning are configuration (serving-manifest fields,
layout files read at run time, model metadata). Nothing is a predefined setup in code, nothing needs a separate build
per model or configuration, and no code is duplicated per model. Upstream Radiance is the baseline; a patch only adds or
extends capability, generically, and defaults to upstream behaviour.

A static, read-only audit of branch `feat/vtp14-n1-rank-code-eligibility` (GPT reviewer through the agents gateway, at
`91d527201`..`85a0abf07`) found the violations below. Status is kept here as they are fixed.

## Already done on this branch

- `--tp-ownership SPEC` is a generic split checked against the loaded model; no model or profile in the patches.
- The pipelined prefill is a mode and a narrowest step, not fixed world and row gates.
- Layouts are run-time data (`--layouts` / `BIGCHERRY_LAYOUTS`), not compiled into `bc-serve`; `layout.name` has no default.
- A layout's engine may be several `{ arch, count }` groups (a mixed world); a layout's `[settings]` table holds resolver keys.
- Contract amendment 18: `tuning.tp_ownership`, `tuning.pipelined_prefill(_min_rows)`, `device.rank_devices`.
- One build: the VTP14 patches are a patch set in the main `radiance` source, inert until their options are given
  (ABBA in VTP14.md).

## Findings

| ID | Impact | What is hardcoded or duplicated | Where | Generic replacement | Status |
|---|---|---|---|---|---|
| C01 | critical | remote-MoE shared-memory offsets from `kHidden = 2560`, `kTopK = 10`, `kExperts = 512` | `moe-remote/moe_remote_shm.h`, `moe_mailbox_ops.hip`, `moe_remote_gpu.h` | the negotiated `LayoutShape` (worker_shape.h) and its computed offsets everywhere; container-derived dims | open |
| C02 | high | remote-MoE and gfx12 GEMV refuse `K > 2560` (fixed LDS arrays) | `moe-remote/moe_lane_gemm.h`, `bc-gfx12/bc_moe_gemv.hip` | shape-independent tiling, or dispatch to the generic GEMM | open |
| C03 | high | MTP pass replay accepts head dim 256, GQA 6/12, KV block 4; reproduces the split-KV calculation | `patches/rad0013_mtp_pass_replay` | ask the attention planner for a replay-safe geometry | open |
| C04 | critical | Python inventory adapter refused a mixed-architecture engine | `engine_adapters/radiance_inventory.py` | validate each rank; keep `rank_devices` order | **fixed** (4246c3ead) |
| C05 | high | remote-MoE topology bounds compiled in (2 engine ranks, 4 helpers, 4 slots) | `moe-remote/moe_remote_shm.h` | size protocol tables and slots from validated run-time counts | open |
| C06 | medium-high | `archs.toml` compiled into `bc-serve` | `serving/CMakeLists.txt`, `serving.cpp` | load card-type descriptors at run time, digest in the manifest | open |
| C07 | high | launcher checked one comma-joined plugin file for a mixed world | `lab/radiance/launch-flashnext-best.sh` | check each engine plugin | **fixed** (4246c3ead) |
| C08 | high | n-gram GPU placement has no manifest field (`--ngram-device` refused) | launcher, `serving-v2.toml` | a typed device selector by BDF, bound to radiance's option | open |
| C09 | high | rad0008 defaults `RAD_HC_UPMIX_RT` to 128 (upstream 256) with no manifest field | `patches/rad0008_hc_upmix_rt_sweep` | upstream default when unset; a typed `tuning.hc_upmix_rt` | open (changes Flash-Next serving: needs an A/B) |
| C10 | high | routing-heat scratch `step_heat[kExperts]` caps experts at 512 | `moe-remote/moe_mailbox_ops.hip` | size from the container's expert count | open |
| C11 | medium-high | layouts and recipe runtime profiles carry raw `server-args`; Python allowlist and shell mapping repeat them | `layouts/*.toml`, `recipes.toml`, `radiance_layout.py`, launcher | typed service/execution/memory/tuning fields; one binding from `serving-v2.toml` | open |
| C12 | medium-high | Python accepts any `R3_MOE_WORKER_*` layout env; C++ accepts only `R3_MOE_WORKER_COMBINE` | `radiance_layout.py`, `serving.cpp` | typed helper settings (`helper.combine`), one validator | **fixed**: a layout's `[settings]` takes resolver keys (`helper.combine`, `tuning.*`); `env` and `tuning` tables are refused by both resolvers |
| C13 | medium-high | diagnostics need separate sources and a `-DBC_MOE_REMOTE_DIAG` build | `recipes.toml`, `patches/rad0004` | diagnostic hooks in the common build behind typed `diagnostics.*` (keep `radiance-native` as the control) | open |
| C14 | medium | calibration and helper-profile seeds compiled in; seeds match `flashnext-4gpu` | `serving/CMakeLists.txt`, `calibration.toml` | run-time records with provenance | open |
| C15 | medium | Python layout resolver duplicates the C++ one (C04 and C12 are real divergences) | `radiance_layout.py`, `radiance_inventory.py`, `serving.cpp` | Python consumes `bc-serve --explain --json` only | open |
| C16 | medium | launcher defaults to `flashnext-4gpu`, a Flash-Next heat file and run dir | `launch-flashnext-best.sh` | require a layout or manifest; heat from `handoff.heat` | open |
| C17 | medium | lab tools name layouts (`LAYOUTS` dict), a tail count 5931, a literal BDF | `rhp20-forced-move.py`, `rhp01-coverage.sh`, `heat-watch.py`, `helper-capture.sh` | layouts and experiment files as inputs; values from owner maps and inventory | open |
| C18 | medium | ownership, delta-net head and vocabulary-span edits repeated per plugin (bf16, fp8) | `patches/rad0018`, `rad0019`, `rad0020` | shared architecture operations; plugins declare capability | open |
| C19 | medium | v1 resolver defaults from one deployment (context 262144, pool 9216, host KV 12288, depth 3) | `serving.cpp` | container metadata and measurement | open |
| C20 | low (dormant) | n-gram helper requires `n_embd = 160`, `n_vocab = 320001536` | `ngram-helper/r3_ngram_host.cpp` | model-derived sizes, or retire the plugin | open |
| C21 | low-medium | power-plan check branches on `arch == "gfx1100"` | `serving/power_plan.py` | a typed card-type power capability | open |
| C22 | conditional | llama.cpp 1217 turns graph optimisation on by default on RDNA3.5 | `engines/llamacpp/patches/1217_*` | upstream default unless opted in | open (selection in production not confirmed) |
| C23 | critical | the P1 producer refused every container without routed experts: no dense layout could write a measured manifest | `serving/v2/p1_producer.cpp` | experts are read only for helpers; engine-only needs none | **fixed** (a14900fd0; found by the TP3 launcher run) |
| C24 | critical | the retention target had to be positive everywhere (contract `ref positive`), so an engine-only layout without a host KV tier could not write a measured manifest | P1 producer, P1/native measurement, S8, `serving-v2.toml` | 0 tokens means no host KV tier (amendment 19) | **fixed** (613a612f1) |

The model-constant guard (`engines/radiance/model_constant_guard.toml`) scans Flash-Next literals only. It misses the
head-dim/GQA eligibility of C03, the 5931 tail count of C17, topology bounds, named defaults and duplicated
flag mappings; extend it with C01-C05/C10 as owners close them.

## Plan

- **P0, model-derived execution geometry** (C01, C02, C05, C10): one negotiated shape and checked allocation for the
  remote-MoE protocol, heat scratch and GEMV widths. Flash-Next's helper path; kernel work.
- **P1, mixed-architecture end to end** (C04, C07 done; C08): the n-gram device selector.
- **P2, the typed manifest as the one authority** (C11, C12, C14, C15, C19): retire raw `server-args` and layout env,
  Python consumes the C++ resolution, run-time policy data.
- **P3, upstream defaults and shared capability** (C09, C03, C22, C18, C13).
- **P4, configuration-driven lab tooling and a stronger guard** (C16, C17, C20, C21).
