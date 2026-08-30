# Reference — by concern

Organized into topical subfolders. Cross-cutting/auto-generated docs stay
at the top level.

## Architecture

| Document | What it is | When to read it |
| --- | --- | --- |
| [architecture/OVERVIEW.md](architecture/OVERVIEW.md) | Phases, status, principles, verification | Understanding scope and progress |
| [architecture/MULTI_GPU_DISPATCH.md](architecture/MULTI_GPU_DISPATCH.md) | Layer split vs tensor split; RCCL/internal/META reduction provider selection and fail-closed policy | Understanding what runs on multi-GPU, or debugging a heterogeneous-topology issue |
| [architecture/FAMILY_MODEL.md](architecture/FAMILY_MODEL.md) | Six families, identity rules, rejected proposals | Designing a new family or candidate identity |
| [architecture/DESIGN_DECISIONS.md](architecture/DESIGN_DECISIONS.md) | Architecture decisions + operational gotchas | Understanding why something is shaped a certain way; avoiding traps |

## Build

| Document | What it is | When to read it |
| --- | --- | --- |
| [build/BUILD.md](build/BUILD.md) | Environment, recipes, cmake configuration | Building on hardware |
| [build/PIN_BUMP.md](build/PIN_BUMP.md) | Bumping the pinned llama.cpp revision | Moving the pin forward |

## Testing

| Document | What it is | When to read it |
| --- | --- | --- |
| [testing/TEST.md](testing/TEST.md) | Test invocations, dispatch modes, tuning/coverage workflows | Testing, tuning, or running on hardware |
| [testing/COVERAGE_AUDIT.md](testing/COVERAGE_AUDIT.md) | What the tuner can/cannot choose between; gaps | Understanding why a signature has few options |

## Tooling

| Document | What it is | When to read it |
| --- | --- | --- |
| [tooling/TOOLING.md](tooling/TOOLING.md) | Domain map: which package owns what, migration state | Finding where a capability lives before adding a new one |
| [tooling/TUNE_CAMPAIGN.md](tooling/TUNE_CAMPAIGN.md) | `bigcherry tune-campaign` — record→tune→correctness→promote→replay orchestrator | Running the full tuning pipeline in one command |
| [tooling/PROFILING.md](tooling/PROFILING.md) | `bigcherry profile-campaign` — real rocprofv3 kernel/timing/resource profiling | Deep-diving why a workload spends time where it does |

## Patches

| Document | What it is | When to read it |
| --- | --- | --- |
| [patches/PATCH_SYSTEM.md](patches/PATCH_SYSTEM.md) | Patch catalog, states, groups, composition | Understanding how patches apply and compose |
| [patches/PATCH_AUTHORING.md](patches/PATCH_AUTHORING.md) | Writing a new patch | Authoring a patch |
| [patches/PATCH_VALIDATION.md](patches/PATCH_VALIDATION.md) | Validation states and evidence requirements | Promoting a patch toward validated |
| [patches/PATCH_REFACTOR_RUNBOOK.md](patches/PATCH_REFACTOR_RUNBOOK.md) | Refactoring an existing patch | Restructuring a patch without breaking its evidence |

Patch-specific documentation, fixtures, validators, and validation evidence
live with the owning package under `patches/<patch-id>/`. The shared
`patches/_validation/` directory contains only cross-patch baseline data; it is
not a reference-document store.

Plan-item design, status, and decisions belong under the matching
`docs/planning/<state>/<plan>/` directory. Campaign run bundles and generated
measurement outputs belong under `artifacts/<campaign-id>/`; retain only a
concise, broadly reusable decision or index in `docs/reference/experiments/`.
Historical reviews, handovers, imported planning sources, and superseded
snapshots belong under [`docs/archive/`](../archive/), outside this maintained
reference corpus. The old `docs/reference/archive/` paths are compatibility
redirects only where historical plan links still require them.

## Experiments

| Document | What it is | When to read it |
| --- | --- | --- |
| [experiments/EXPERIMENTAL_WORKFLOW.md](experiments/EXPERIMENTAL_WORKFLOW.md) | Repeatable diagnostic qualification, run-bundle, promotion, and archival workflow | Before turning exploratory work into a formal plan or campaign |
| [experiments/EXPERIMENT_CONTRACT.md](experiments/EXPERIMENT_CONTRACT.md) | Current experiment-contract schema, identity, and CLI | Defining or consuming an experiment contract |
| [../evidence/2026-08-21-hi35-hi36-27b-r9700/HI36A_VERDICT_27B_R9700.md](../evidence/2026-08-21-hi35-hi36-27b-r9700/HI36A_VERDICT_27B_R9700.md) | HI36A's verdict on the 27B/R9700 case | Reviewing that specific experiment's outcome |

## Top level

| Document | What it is | When to read it |
| --- | --- | --- |
| [START_HERE.md](START_HERE.md) | Stable project orientation and safe first-work checklist for new agents | Starting a fresh agent context or handing work to another agent |
| [CANDIDATES.md](CANDIDATES.md) | Auto-generated candidate inventory | Looking up a specific candidate's config |
| [FINDINGS.md](FINDINGS.md) | Bugs and results notable outside bigcherry — kernel/llama.cpp/ROCm bugs, exceptional tunes | You just found something a stranger to this project would want to know about |
| [TUNING-DETAIL.md](TUNING-DETAIL.md) | Pointer to archived generated report | Consult the tuning DB or `artifacts/reports/` for current numbers |

**Tuning results and model-specific data live in the database**, not in
these documents. Consult the tuning DB for per-signature winners, rankings,
improvements, and coverage numbers.

Historical originals are in [`docs/archive/`](../archive/) for provenance only;
they are not current implementation or status guidance.
