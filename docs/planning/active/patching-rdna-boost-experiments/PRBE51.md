---
id: PRBE51
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:01.483332+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-HIP-004: Full GPU input-layer offload on AMD UMA

## Description

TODO. Full GPU input-layer offload for AMD UMA/iGPU (gfx1151, Strix Halo). Relevance at b11126: upstream already has an integrated-GPU device class (`GGML_BACKEND_DEVICE_TYPE_IGPU`, ggml/include/ggml-backend.h:140) with dedicated handling in device enumeration (src/llama.cpp:260, igpu dedup workaround) and `llama_supports_gpu_offload` (src/llama.cpp:107). gfx1150/1151/1152/1153 are recognized HIP compile targets (ggml/src/ggml-cuda/vendors/hip.h:223) and referenced in CI as having a known batched-inference issue. However, input-LAYER-specific placement policy (keeping input tensors GPU-resident on UMA vs the general host-buffer-list logic in `make_cpu_buft_list`, src/llama-model.cpp:1060-1090) has not been located -- that function adds a host buffer type generically for any device list, with no IGPU-specific carve-out found. Item's premise (repeated CPU/GPU sync from input tensors not staying GPU-resident on UMA) is plausible and not yet addressed by a dedicated policy; TODO stands.

## Steps

1. Grep for the actual input-tensor buffer-type selection call site: `git -C work/upstream/llama.cpp.git grep -n 'buft_list\|input.*buft\|ggml_backend_dev_host_buffer_type' b11126 -- src/llama-model.cpp` and read the full `make_cpu_buft_list` function (src/llama-model.cpp, starts ~line 1060) plus its caller to see how/when the host buffer type is actually assigned to input-layer tensors (vs other layers).
2. Grep for existing IGPU special-casing: `git -C work/upstream/llama.cpp.git grep -n 'GGML_BACKEND_DEVICE_TYPE_IGPU' b11126 -- src/` (3 hits already found: ggml-backend.h:140 enum, llama.cpp:107, llama.cpp:260 -- read llama.cpp:260's surrounding device-enumeration function fully to understand how an IGPU device currently gets treated relative to a discrete GPU for buffer-type list construction).
3. Design a policy gated on `ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU`: when the model's primary compute device is IGPU, skip adding the generic CPU host-buffer-type entry for input-layer tensors specifically (leave it in place for all other UMA vs discrete-GPU behavior), so those tensors resolve to the IGPU's own device buffer type and stay GPU-resident.
4. Instrument a sync-count counter (increment at each host<->device copy triggered by input-tensor access) to measure before/after.
5. Add a correctness test: output parity (temp-0 identity) for a small model run entirely through the new IGPU input-placement path vs the existing host-buffer path, plus a discrete-GPU (XTX/R9700) control confirming zero behavior change there (predicate must be false for non-IGPU devices).
6. Benchmark PP across model/batch sizes on gfx1151 with full offload vs CPU-offload vs discrete-GPU controls; report sync count, PP, and residency.

## Detailed Solution & Technical Design

Hardware-gated (IGPU device-type only) change to input-layer buffer-type selection in `make_cpu_buft_list` / its caller in src/llama-model.cpp, keyed off the already-upstream `GGML_BACKEND_DEVICE_TYPE_IGPU` classification rather than inventing new hardware detection. Discrete-GPU and CPU-offload code paths must be provably untouched (same buft_list construction) when the primary device is not IGPU. This is distinct from PRBE17 (host-buffer async correctness) per the item's own note -- do not conflate the two.

## Code Samples & Guidance

Real anchor (verified via `git -C work/upstream/llama.cpp.git show b11126:src/llama-model.cpp`, lines 1060-1090):
```cpp
// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<llama_device> & devices, bool use_extra_bufts, bool no_host) {
    ...
    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    if (!no_host) {
        for (const auto & dev : devices) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev.dev);
            if (buft) {
                buft_list.emplace_back(dev.dev, buft);
                break;
            }
        }
    }
    ...
```
and IGPU classification (ggml/include/ggml-backend.h:139-140):
```cpp
        // integrated GPU device using host memory
        GGML_BACKEND_DEVICE_TYPE_IGPU,
```
patches/<order>_prbe51_uma_input_offload/patch.toml:
```toml
schema = 1
id = "<order>_prbe51_uma_input_offload"
order = <next available>
state = "untested"
kind = "enhancement"
origin = "local"
backend = "hip"
plan-ids = ["PRBE51"]
requires = []
conflicts = []
requires-options = []
forbids-options = []
subsystems = ["model-loading", "buffer-placement"]
hardware = ["amd", "gfx1151"]
validation-architectures = ["gfx1151"]
backends = ["hip"]
```
patch.py skeleton:
```python
from bigcherry.patcher import Edit, FilePatch

PATCHES = [
    FilePatch(
        path="src/llama-model.cpp",
        description="PRBE51 IGPU-gated input-layer GPU-resident placement, no_host carve-out",
        edits=(
            Edit(
                id="prbe51-igpu-input-placement",
                anchor=r"<TODO-VERIFY: exact text of make_cpu_buft_list's caller where input-layer tensors specifically select their buft, not the generic no_host branch above -- must be read in full first, step 1>",
                rationale="skip the generic host-buffer entry for input-layer tensors when the primary device is IGPU, keeping them GPU-resident",
                mode="replace",
                text=r"<TODO-VERIFY: IGPU-gated branch + BIGCHERRY_PATCH_HIT marker>",
                guard=r"BIGCHERRY_PATCH_HIT patch=prbe51",
            ),
        ),
    ),
]
```

## Files

src/llama-model.cpp; src/llama.cpp (reference, IGPU enumeration); tests/ (output-parity + sync-count instrumentation); patches/<order>_prbe51_uma_input_offload/{patch.toml,patch.py,SUMMARY.md}

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay <id> --source bigcherry-tuning`. Output-parity temp-0 identity unit test. Hardware (Brutus/Strix-Halo host, not run here): gfx1151 full-offload vs CPU-offload vs discrete XTX/R9700 controls across prefill batches/model sizes, sync count, PP, residency, via `python -m bigcherry.patch.validation_campaign`.

## Effort & Risk

M / medium -- touches core buffer-type selection in model loading; risk contained by strict IGPU-device-type gating (discrete-GPU/CPU paths must be provably unchanged).

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Promote only for verified UMA/iGPU hardware with output parity, GPU residency, reduced synchronization, and repeatable prefill gain; never generalize to discrete GPUs without independent evidence.

## Notes

Supersedes: RD61
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd61

2026-09-24 relevance at b11126: GGML_BACKEND_DEVICE_TYPE_IGPU device class and gfx1150/1151/1152/1153 HIP targets already upstream (ggml-backend.h:140, hip.h:223, llama.cpp:107/260), but no input-layer-specific GPU-residency policy found in make_cpu_buft_list (src/llama-model.cpp:1060-1090) or its caller -- TODO stands. GPT design request req_59325a19cc8d4adb (batched with PRBE56, submitted, response pending as of this pass -- written directly from source evidence, to cross-check against GPT response once available).

## Change Log

- 2026-09-09T10:57:01.483332+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:15.456287+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.359287+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.144228+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:11:46.041972+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031217_repaired-four-more-active-succ_7909
- 2026-09-10T03:12:17.932869+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:34:36.673061+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
