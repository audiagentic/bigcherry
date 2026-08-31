# Experiment contracts

Experiment contracts describe an external or experimental optimisation in a
form that BigCherry can validate and expand into campaign lanes. They are
configuration and evaluation metadata; they do not define a new kernel
family, dispatch key, benchmark framework, or runtime model dispatcher.

## Source of truth

- Registry: `config/experiment-contracts.toml`
- Schema, parsing, hashing, gates, and promotion helpers:
  `tools/bigcherry/experiment/contract.py`
- Real per-lane paired execution: `tools/bigcherry/experiment/execution.py`
  (VA14 — see
  [PATCH_VALIDATION.md's "Real contract-execution architecture"](../testing/PATCH_VALIDATION.md#real-contract-execution-architecture-va14)
  for the full picture, including `patch/validation_campaign.py`'s
  validation-domain build/lane wiring). `tools/bigcherry/campaign/` provides
  shared statistics/environment primitives (`block_bootstrap_effect()`,
  `sanitize_environment()`) that `execution.py` reuses — it is not itself
  the contract executor.
- Current work and acceptance state: the matching item under
  `docs/planning/active/` or `docs/planning/completed/`

The registry contains one `[contract.<id>]` table per contract. Source IDs
must resolve through `config/external-sources.toml` when callers request that
cross-check. Contract IDs and hashes are provenance; they must never become
part of a runtime dispatch signature or candidate name.

## Contract shape

Every contract declares:

```toml
[contract.EXAMPLE]
title = "Short, stable description"
prerequisites = []

[contract.EXAMPLE.source]
source_id = "external-source-id"
commits = ["immutable commit SHA"]
atomic_part = "independently testable transform"

[contract.EXAMPLE.hypothesis]
family = "mmvq"                 # one of the existing runtime families
expected_effect = "both"        # performance, correctness, or both
rationale = "Why this should change the measured workload"

[contract.EXAMPLE.scope]
backend = "hip"
architectures = ["gfx1100"]
weight_types = ["q6_k"]

[contract.EXAMPLE.positive]
models = ["model-or-recipe-ref"]
workloads = ["decode"]

[contract.EXAMPLE.controls]
models = ["model-or-recipe-ref"]
workloads = ["prefill"]

[contract.EXAMPLE.boundary.dimensions]
physical_m = [1, 2, 4, 8]

[contract.EXAMPLE.correctness]
bit_identical = "required"

[contract.EXAMPLE.acceptance]
target_kernel_gain_pct = 1.0
end_to_end_gain_pct = 0.0
max_control_regression_pct = 1.0
```

The parser validates known workload tags, kernel-family names, thresholds,
source linkage, prerequisites, and prerequisite cycles. Optional scope fields
include integrated/UMA/peer-access requirements, GPU-count bounds, and driver
version bounds. Correctness checks are drawn from the closed set
`backend_reference`, `greedy_parity`, `bit_identical`, `ppl_equality`, and
`state_restore_integrity`.

`state_restore_integrity` (VA10) names an affirmative invariant --
saved state → repeated multi-GPU restore → restored state/continuation
agrees with reference -- for patches whose claim is about correctness of a
save/restore cycle rather than kernel output parity. It is deliberately
NOT an absence-of-fault claim: failing to observe a specific fault again is
not proof the fault is fixed, especially when that fault was itself hard to
reproduce in the first place. Pair it with the `state_restore` workload tag
(distinct from the generic `multi_gpu_copy` transfer-workload tag -- a
save/restore cycle is not simply a copy).

### Resource-cost acceptance (VA12)

Some patches trade a timing win for a resource-cost risk -- e.g. RD73's
stable graph-cache key retains more shape-specific cache entries, which can
win on timing while quietly growing memory/entry-count unboundedly.
`acceptance.resource_limits` declares that budget as an array of tables,
additive to the three existing scalar acceptance fields:

```toml
[[contract.EXAMPLE.acceptance.resource_limits]]
metric = "graph_cache_entries"
unit = "count"
max_value = 32

[[contract.EXAMPLE.acceptance.resource_limits]]
metric = "graph_cache_resident_bytes"
unit = "bytes"
max_increase_pct = 5.0
```

`metric` is an open, non-empty identifier (like `source-evidence.metric` --
resource kinds genuinely vary and a fixed vocabulary would either lose
precision or invite a wrong-but-close mapping). `unit` is closed to
`bytes`/`count` -- a dimensional mismatch is a real, dangerous class of
error. At least one of `max_value` (an absolute ceiling on the subject's
measured value) or `max_increase_pct` (a bound on growth over the paired
control) must be declared; both may be declared together as independent
checks. Duplicate metrics within one contract are rejected.

Evidence is supplied as a `ResourceResult` (metric, unit, subject_value,
optional control_value) and checked by `evaluate_resource_gate()`, which is
fail-closed the same way `evaluate_correctness_gate()` is: missing evidence
for a declared limit is a FAIL, not "not applicable"; a `max_increase_pct`
limit with no `control_value` evidence is a FAIL (the bound cannot be
evaluated); a `control_value` of 0 paired with any positive `subject_value`
is explicit unbounded relative growth and FAILS rather than dividing by
zero or silently passing. A contract that declares `resource_limits` at all
blocks promotion (`evaluate_promotion_gate()`) unless a passing
`resource_gate` is supplied -- exactly like a missing/failed correctness
check. A contract with no `resource_limits` is unaffected.

`resource_limits` participates in `contract_hash` only when non-empty, so
every contract written before VA12 keeps its exact original hash.

## Identity and evidence rules

Keep these identities separate:

1. the external source or idea;
2. the atomic BigCherry transform;
3. the runtime candidates exposed by that transform.

Use captured canonical signatures and workload metadata rather than inventing
model-name or hand-authored shape keys. Contract evidence travels alongside
the normal provenance document and includes the contract ID, immutable hash,
optimization ID, lane role, workload, model, and boundary value. It must not
alter dispatch or build identity.

Promotion requires the declared correctness evidence, positive performance,
control-regression limits, boundary coverage, provenance, and any applicable
holdout/generalisation proof. A contract report renders evidence already
collected; it does not manufacture measurements.

## CLI

Run from the repository root with the normal tooling path:

```bash
PYTHONPATH=tools python -m bigcherry experiment-contract validate
PYTHONPATH=tools python -m bigcherry experiment-contract list
PYTHONPATH=tools python -m bigcherry experiment-contract plan \
  --source <source> --build <build> --platform <platform> <contract-id>
PYTHONPATH=tools python -m bigcherry experiment-contract run \
  --source <source> --build <build> --platform <platform> <contract-id>
PYTHONPATH=tools python -m bigcherry experiment-contract report \
  --evidence-file <evidence.json> <contract-id>
```

`validate`, `list`, and `plan` are offline or dry-run operations. `run`
materialises and executes the contract's campaign lanes; it does not replace
the correctness, comparison, or promotion stages. `report` renders a stored
evidence document and returns failure when the promotion gate fails.

When adding a contract, update its owning plan item and add schema/identity,
negative-case, lane-expansion, evidence, and promotion-gate tests. Keep
historical design discussions in `docs/archive/`, not in this reference.
