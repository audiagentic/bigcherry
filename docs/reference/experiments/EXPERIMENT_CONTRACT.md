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
  [PATCH_VALIDATION.md's "Real contract-execution architecture"](../testing/PATCH_VALIDATION.md#real-contract-execution-architecture)
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

## Lanes: a control may not be the treatment

A lane is identified by `(model, workload)` and nothing else -- an
`EvaluationSet` carries no further axis. The same lane must therefore never
appear in both `positive` and `controls`. `parse_contract` rejects it.

Such a contract asks one measurement to satisfy two contradictory
requirements at once: gain at least `target_kernel_gain_pct`, AND change by no
more than `max_control_regression_pct`. Worse, it makes the regression budget
self-referential. A control exists to detect collateral damage on work the
hypothesis does *not* claim to speed up; if the control IS the treatment,
there is no such work being watched and the budget is decorative.

The rule is per-lane, not per-set. Sharing a model across roles is fine, and
"the same workload on a model the hypothesis does not claim" is the standard,
correct control pattern -- it is explicitly permitted.

Choose a control by asking what the change touches without claiming. For a
kernel selected only on decode-shaped matmuls, prefill is the control. For a
default flip that alters the whole graph while claiming decode, prefill is
again the control. For a per-head-dimension attention configuration, a model
with different attention geometry is the only control that can detect a
mis-selected config.

## Outlier handling: frozen policy

**There is none, and this is deliberate.** `block_bootstrap_effect()` keeps
every valid pair. No trimming, no winsorizing, no MAD/IQR/sigma test, and no
threshold on the treatment/control ratio.

When one extreme pair widens the interval past the bound, that is the correct
result: the run is *uninformative*, and `ci95_threshold_bound_v1` refuses
promotion rather than reporting a number nobody should act on. Do not make the
estimator robust merely because environmental stalls exist.

Freeze this distinction:

- **Validity failure** -- evidence that the measurement or protocol did not
  execute as specified, defined *independently of any expected treatment
  benefit*. Legitimate examples: the benchmark process crashed or timed out
  abnormally; a GPU reset or fault was recorded; required telemetry is missing
  or corrupt; unrelated load exceeded a pre-registered bound; clock/power/
  thermal state fell outside a pre-registered operating envelope; benchmark
  output failed its correctness check; an external timer disagrees with the
  benchmark timer beyond calibrated tolerance. Prefer arm-local or external
  health signals, evaluated *without* comparing the two arms.
- **Extreme observation** -- a valid measurement with an unusual effect.
  **Keep it.**

Never use a statistical outlier test to move an observation from the second
category into the first.

### The predicate that looks principled and is not

A tempting rule, considered and **rejected**:

> the two arms show identical `draft_acceptance`, yet wall-clock differs by
> more than Y, therefore the pair is instrumental and is discarded

Equal work does not establish an instrumental fault. A patch can produce
identical speculative acceptance while making that same work dramatically
slower through kernels, synchronisation, memory traffic, communication, or
scheduling -- which is precisely the regression class a promotion gate exists
to catch. The predicate conditions directly on the effect being estimated, so
it can delete genuine regressions. A threshold on the treatment/control timing
ratio is presumptively illegitimate in a performance experiment for exactly
this reason: it truncates the distribution it is measuring.

Outcome-based rejection is defensible only for demonstrable sensor
impossibility or corruption, never for "this value is surprising".

### Re-running

Never do this:

    interval too wide -> discard the run -> collect a fresh one

Do this instead: pre-declare `N_min` and `N_max`; if the precision criterion is
not met, collect further deterministic pairs up to `N_max` and estimate over
**all** valid pairs. The precision criterion must be **direction-blind** --
interval *width*, never `ci95_low` relative to the threshold. A run found
genuinely invalid may be repeated, but its invalid evidence stays recorded.

This is implemented, not just doctrine, by
`session_ci95_threshold_bound_v1` (`min_sessions` / `max_sessions` /
`max_ci95_width_pct`). `_session_stopping_rule_met()` is direction-blind *by
construction*: it is never passed the acceptance threshold, so it cannot stop
early because the answer looks good.

RD73 is the worked example, and it earned the rule. The stopping rule twice
refused evidence that a plain `ci95_low >= threshold` gate would have passed:

| sessions | `ci95_low` (bar 1.0) | width (target 1.0) | verdict |
|---|---|---|---|
| 4 | 1.2929 | 1.111 | INCONCLUSIVE |
| 5 | 1.4289 | **1.0101** | INCONCLUSIVE |
| 6 | 1.4754 | 0.8769 | **PASS** |

The five-session refusal missed the target by 0.0101 -- the exact moment a
movable threshold would have been moved. Because the criterion had been
committed before the evidence, it was not. Note also that the loop stopped the
instant the rule decided: **continuing past a decision is optional stopping in
the other direction.**

### Sessions, not just pairs

A paired interval from one run covers only within-session variation.
Measured on real hardware, six sessions of the same unchanged RD73 build gave
a between-session sd of ~0.52 -- larger than the standard error any single run
reported, with one session's point estimate below another's `ci95_low`.

So where a claim must hold across occasions, the session is the unit:
`bootstrap_session_effect()` resamples whole sessions (a cluster bootstrap,
minimum `MIN_BOOTSTRAP_SESSIONS = 4`) and `aggregate_session_effects()`
rebuilds the interval from the `lane_effects` persisted in each validation
record.

Resampling *session* identity is correct precisely where resampling *lane*
identity is not: lanes are fixed components a contract names by hand (decode,
prefill), so "sampling a lane" is meaningless; sessions are exchangeable draws
from the occasions a measurement could have been taken on, which is what a
claim generalises over.

### Regression and improvement are not symmetric

`improvement_no_regression_v1` drops the materiality bar entirely: an
improvement must be **established** (`ci95_low` above `min_evidence_effect_pct`)
but need not reach any size, while the regression budget stays strictly
interval-bounded. Declaring a gain bar under it is rejected at parse time.

The rationale is a deliberate asymmetry of costs, not leniency: shipping a
regression costs real throughput, whereas adopting a genuine small improvement
costs little beyond the patch's own maintenance. The earlier policies conflate
"is the effect real" (evidence) with "is it worth carrying" (a value
judgement); this one keeps only the first.

Two things it does **not** relax:

- **Established means the interval.** `+1.95%` with CI `[-0.10, 4.0]` fails.
  A positive point estimate whose interval reaches the floor is not a win.
- **`min_sessions` still applies.** Drift does not stop being real because
  the decision rule changed.

`min_evidence_effect_pct` is where a *measured harness bias floor* belongs. It
defaults to 0.0, but if the rig systematically reads +0.2% for reasons
unrelated to any patch, a 0.0 floor would admit an unbounded stream of
"established" wins that are measurement artifacts -- and they would look more
significant the more sessions were collected. That constant should come from
an A-A null run (control vs control, identical builds). **No such run exists
yet**, so 0.0 is a placeholder, and adopting sub-1% gains under this policy is
only as sound as that assumption.

### Governing a change to this policy

If validity exclusion, trimming, or a dual-estimator gate is ever added:

1. Do **not** mutate `*_v1`. Introduce a new versioned policy ID; if validity
   is orthogonal to interval acceptance, it belongs in a separately versioned
   `measurement_validity_policy`, not bolted onto the effect policy.
2. Policy and constants are committed *before* qualifying evidence is
   collected.
3. Existing evidence stays tagged with the policy in force when it was
   collected.
4. A new policy is prospective for promotion. Historical runs may be used to
   validate or simulate it, but must not be selectively reinterpreted to
   promote the patch that motivated the change.
5. Any constant change -- trim fraction, health threshold, `N_max` -- is a new
   policy version.
6. Constants are chosen from unrelated calibration data, never from outcomes
   of patches awaiting promotion.

(Frozen after adversarial review by dev-gpt-agent, `req_6eea1c280d824cee`.)

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

