---
id: PRBE61
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:45.405833+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# VK-TUNE-001: Vulkan rm_kq specialization sweep

## Description

TODO. Sweep Vulkan mul_mat_vec rm_kq (rows-per-shader-invocation for K-quant types) per driver/architecture/kernel-signature instead of the current fixed constant. Relevance at b11126: confirmed rm_kq is still a compile-time-ish constant set once at device init (ggml-vulkan.cpp ~2676-2697): default 2, bumped to 4 only for `AMD_GCN`; RDNA3/RDNA4 (our gfx1100/gfx1201 hardware) fall through to the default rm_kq=2 with only a separate, unrelated `is_rdna3` static-4-rows override applied to a different code path (rm_int_n/rm_id lambdas, not rm_kq itself). No per-driver or per-signature table exists -- item's premise holds, not upstream-absorbed.

TODO, corrected mechanism. GPT identified a real defect: rm_kq is a single scalar consumed by ALL K-quant/TQ pipeline-creation call sites (Q2_K..Q6_K, TQ1_0), and `rm_iq = 2 * rm_kq` (verified at ggml-vulkan.cpp:2697) derives the IQ-type row count FROM rm_kq -- a per-(driver,arch,quant) OVERRIDE TABLE cannot simply reassign the single `rm_kq` variable after the fact, because that would unintentionally also change every IQ-type pipeline's rm_iq value. The override must instead be a per-type function/table (`rm_kq_for(ggml_type)` or `rm_kq_by_type[GGML_TYPE_COUNT]`) consulted separately at each K/TQ pipeline-creation call site, computed AFTER vendor/architecture selection, leaving the baseline scalar `rm_kq`/`rm_iq` (and therefore every IQ pipeline) completely untouched.

## Steps

1. Re-confirm the anchor at qualification time: `git -C work/upstream/llama.cpp.git grep -n 'uint32_t rm_kq' b11126 -- ggml/src/ggml-vulkan/ggml-vulkan.cpp`.
2. Benchmark rm_kq in {1,2,3,4} on R9700 (RDNA4/gfx1201) across RADV and AMD proprietary/AMDVLK, plus gfx1100 (RDNA3) as a control, for each K-quant pipeline that consumes rm_kq (Q2_K..Q6_K, TQ1_0 mul_mat_vec pipelines registered at ggml-vulkan.cpp:2746-2751) -- inspect VGPR/register usage (via driver shader-stats dump where available) and kernel time, plus end-to-end PP/TG.
3. Retain only repeatable per-(driver, architecture, quant-type) winners; do not promote a single new global constant (that would repeat the same mistake the item is fixing).
4. Implement an opt-in override table keyed by (vendor, architecture enum, driver id/string, quant type) inserted immediately after the existing `rm_kq` assignment block, active only when `BIGCHERRY_VK_RM_KQ_OVERRIDE=1` (qualification-only; no default-on promotion without separate sign-off per the item's acceptance criteria).
5. Add the activation marker at the point rm_kq is consumed by pipeline creation (ggml-vulkan.cpp:2746 onward), gated by BIGCHERRY_PATCH_TRACE, once per process per quant type.
6. Add a test-backend-ops MUL_MAT_VEC correctness pass for each K-quant type at both rm_kq=2 (baseline) and any overridden value, confirming reference-parity output (rm_kq only changes row-batching, never numerics, but the new dispatch path must be exercised).
7. Report VGPR/registers, kernel time, PP/TG vs the unmodified baseline; keep the override table sparse -- only entries with measured, repeatable wins.

1. Re-confirm the anchor: `git -C work/upstream/llama.cpp.git grep -n 'uint32_t rm_kq\|rm_iq = 2' b11126 -- ggml/src/ggml-vulkan/ggml-vulkan.cpp` (confirmed: rm_kq default/AMD_GCN block ~2676-2691, `rm_iq = 2 * rm_kq` at line 2697).
2. Benchmark rm_kq in {1,2,3,4} on R9700 (RDNA4/gfx1201) across RADV and AMD proprietary/AMDVLK, plus gfx1100 (RDNA3) control, for each K-quant pipeline that consumes rm_kq (Q2_K..Q6_K, TQ1_0 -- ggml-vulkan.cpp:2746-2751) -- inspect VGPR/register usage and kernel time, plus end-to-end PP/TG. Use `vk::DriverId::eAmdProprietary` (the exact b11126 enum symbol) for proprietary-driver detection, not a string key.
3. Retain only repeatable per-(driver, architecture, quant-type) winners; materialize ONLY measured winner rows in the override table -- do not fabricate placeholder rows.
4. Implement `rm_kq_for(ggml_type type)` (or `rm_kq_by_type[GGML_TYPE_COUNT]` populated after vendor/arch selection) computed immediately after the existing baseline `rm_kq`/`rm_stdq` block, leaving the baseline `rm_kq` variable and the derived `rm_iq = 2 * rm_kq` line completely unmodified -- IQ pipelines continue to read the unmodified baseline rm_iq.
5. Use the new per-type value ONLY at each K/TQ pipeline-creation call site (replace the bare `rm_kq` argument with `rm_kq_for(GGML_TYPE_Q2_K)` etc. at each relevant call), active only when `BIGCHERRY_VK_RM_KQ_OVERRIDE=1`.
6. Add the activation marker at pipeline creation, gated by BIGCHERRY_PATCH_TRACE, once per process per quant type.
7. Add a test-backend-ops MUL_MAT_VEC correctness pass for each K-quant type at both baseline and overridden rm_kq_for() value, confirming reference-parity output; add a regression check that IQ-type pipelines' rm_iq is byte-identical to baseline regardless of override state.

## Detailed Solution & Technical Design

Opt-in lookup table layered next to the existing (vendor,architecture)-only rm_kq logic, extended with a driver-id and quant-type axis. Default/env-unset path is byte-for-byte the current b11126 selection. This is an init-time device-capability decision, not a runtime toggle, so the gating env var only controls whether the override table is consulted at pipeline-creation time -- no per-inference-call branching.

## Code Samples & Guidance

Real anchor (verified via `git -C work/upstream/llama.cpp.git show b11126:ggml/src/ggml-vulkan/ggml-vulkan.cpp`, lines 2676-2697):
```cpp
    // the number of rows computed per shader depends on GPU model and quant
    uint32_t rm_stdq = 1;
    uint32_t rm_kq = 2;
    uint32_t rm_stdq_int = 1;
    uint32_t rm_kq_int = 1;
    auto const &rm_iq_int = [](uint32_t i) { return i == 0 ? 8u : 4u; };
    if (device->vendor_id == VK_VENDOR_ID_AMD) {
        if (device->architecture == AMD_GCN) {
            rm_stdq = 2;
            rm_kq = 4;
            rm_stdq_int = 4;
        }
    } else if (device->vendor_id == VK_VENDOR_ID_INTEL) {
        rm_stdq = 2;
        rm_stdq_int = 2;
    }
    // RDNA3: above four columns, static 4 rows for all types bench faster than the default
    const bool is_rdna3 = device->vendor_id == VK_VENDOR_ID_AMD && device->architecture == AMD_RDNA3;
```
patches/<order>_prbe61_vk_rm_kq_override/patch.toml:
```toml
schema = 1
id = "<order>_prbe61_vk_rm_kq_override"
order = <next available>
state = "untested"
kind = "enhancement"
origin = "local"
backend = "vulkan"
plan-ids = ["PRBE61"]
requires = []
conflicts = []
requires-options = []
forbids-options = []
subsystems = ["vulkan", "mul-mat-vec", "tuning"]
hardware = ["amd"]
validation-architectures = []
backends = ["vulkan"]
```
patch.py:
```python
import re
from bigcherry.patcher import Edit, FilePatch

_RM_KQ_OLD = """    uint32_t rm_stdq = 1;
    uint32_t rm_kq = 2;"""

_RM_KQ_NEW = """    uint32_t rm_stdq = 1;
    uint32_t rm_kq = 2;

    // bigcherry: PRBE61 qualification-only per-(vendor,arch,driver,type) rm_kq
    // override table. Inactive unless BIGCHERRY_VK_RM_KQ_OVERRIDE=1 -- default
    // path below is byte-identical to b11126 until this env var is set.
    const bool bigcherry_rm_kq_override_enabled = getenv(\"BIGCHERRY_VK_RM_KQ_OVERRIDE\") != nullptr;"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-vulkan/ggml-vulkan.cpp",
        description="PRBE61 opt-in rm_kq override table",
        edits=(
            Edit(
                id="prbe61-rm-kq-flag",
                anchor=re.escape(_RM_KQ_OLD),
                rationale="introduce the qualification-only override gate beside the existing rm_kq default",
                mode="replace",
                text=_RM_KQ_NEW,
                guard=r"BIGCHERRY_VK_RM_KQ_OVERRIDE",
            ),
            Edit(
                id="prbe61-rm-kq-table",
                anchor=r"<TODO-VERIFY: exact text of the is_rdna3 block and rm_iq assignment that follow, to insert the table lookup after vendor/arch is known but before rm_iq = 2 * rm_kq is computed>",
                rationale="apply the measured per-driver/arch/type rm_kq override, sourced from qualification bench results, when enabled",
                mode="replace",
                text=r"<TODO-VERIFY: override table + lookup + activation marker>",
                guard=r"BIGCHERRY_PATCH_HIT patch=prbe61",
            ),
        ),
    ),
]
```

## Files

ggml/src/ggml-vulkan/ggml-vulkan.cpp; tests/test-backend-ops.cpp; patches/<order>_prbe61_vk_rm_kq_override/{patch.toml,patch.py,SUMMARY.md,validation/producer.py}

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay <id> --source bigcherry-tuning`. Unit: MUL_MAT_VEC backend-op parity for Q2_K..Q6_K/TQ1_0 at baseline and overridden rm_kq. Hardware (Brutus, not run here): R9700 RADV vs AMD proprietary/AMDVLK, gfx1100 control, VGPR/register + kernel-time + PP/TG per (driver,arch,type), via `python -m bigcherry.patch.validation_campaign`.

## Effort & Risk

S / low-medium -- init-time constant selection only, no per-call branching; risk is mainly benchmark-noise (false winners) if sample sizes are too small.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Promote only a per-driver/architecture/signature rm_kq choice with repeatable runtime gain and no resource/correctness regression; no global constant.

## Notes

Supersedes: RD78
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd78

2026-09-24 relevance at b11126: confirmed rm_kq is a fixed vendor/architecture-keyed constant (ggml-vulkan.cpp:2676-2697), default 2, only AMD_GCN gets 4; RDNA3/RDNA4 untouched -- no per-driver/signature table exists. GPT design request req_2f372f7cd0914a6f (batched with PRBE49+PRBE50, submitted, response pending as of this pass -- written directly from source evidence, to cross-check against GPT response once available).

2026-09-24 GPT review req_2b65d50ebe9547fd applied: NOT-READY -- corrected mechanism from single-scalar reassignment to a per-type rm_kq_for(type)/rm_kq_by_type table, since rm_iq = 2*rm_kq (verified, line 2697) would otherwise leak the override into unrelated IQ pipelines; specified vk::DriverId::eAmdProprietary as the exact detection symbol; added IQ-non-regression test.

## Change Log

- 2026-09-09T10:57:45.405833+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:55.801381+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.403269+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.219232+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:16:18.025837+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031644_repaired-four-more-active-succ_8062
- 2026-09-10T03:16:44.961564+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:33:47.589745+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:43:57.765144+00:00 (updated-by): Updated: section:description, section:steps
- 2026-09-24T04:44:03.195312+00:00 (updated-by): Updated: section:notes
