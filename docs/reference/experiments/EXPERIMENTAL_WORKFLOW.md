# Experimental qualification workflow

Use this workflow for an exploratory hardware, runtime, toolchain, or
upstream-behaviour question before turning it into a formal implementation
plan. It makes a small experiment repeatable, reviewable, safe, and easy to
archive. It does not authorize production code, dispatch changes, or candidate
promotion.

For an optimisation ready to become a campaign lane, use the
[experiment-contract model](EXPERIMENT_CONTRACT.md). For deep GPU tracing, use
[profile-campaign](../tooling/PROFILING.md). This workflow is the qualification
layer before either becomes appropriate.

## Lifecycle

Record these states in the local brief and final summary:

`proposed -> dry-run -> qualified -> concluded -> promoted | repeated | archived | discarded`

“Qualified” means the harness and controls are trustworthy, not that the
hypothesis is true. “Promoted” means the result justified a formal plan item;
it does not mean code is ready for production.

### 1. Write a qualification brief

For exploratory topics create `tools/lab/<topic>/README.md`. Do not put
plan-specific drivers in `tools/bigcherry/` or add a CLI command yet. State:

- question and falsifiable hypothesis;
- suspected layer (upstream library, driver/toolchain, llama.cpp integration,
  BigCherry, or measurement harness);
- in-scope and explicitly out-of-scope changes;
- topology/devices, software revisions, workload, sizes, and controls;
- one-variable-at-a-time matrix and expected observations;
- safety limits, timeout, stop conditions, and cleanup responsibility;
- result classifications and evidence required for each conclusion;
- owner, run-id convention, and proposed disposition.

Use immutable revision identifiers for source, RCCL/ROCm, binary, model, and
configuration. Diagnostic ordinals, PCI addresses, UUIDs, serials, and
hostnames may be recorded as evidence, but must not silently become future
dispatch identity.

### 2. Dry-run locally

Before touching a GPU or external host, validate command construction,
serialization, case IDs, output paths, timeout, signal, non-zero exit, malformed
output, and partial-run recovery with fake child processes. Run
`bigcherry check --quick` and relevant offline tests. Confirm the experiment
cannot import or mutate production dispatch, tuning, catalog, replay, or
promotion code unintentionally.

For crash-prone collectives, one case must be one child process. Never put a
whole failure matrix in one persistent RCCL process.

### 3. Establish controls

Run the smallest known-good control first, then vary one factor at a time.
Where applicable include homogeneous versus heterogeneous topology, native
versus forced behavior, a known-good size beside a suspected boundary size,
and a negative control that bypasses the suspected subsystem. Use exact
production-relevant tensor sizes. A process staying alive is not correctness
evidence; capture correctness output and completion-synchronized timing
separately.

### 4. Execute with isolation and provenance

Give each run a unique `<run-id>` and each case a deterministic case ID.
Record the exact command, environment, loaded library paths, source/build
revisions, device architecture, topology observations, timestamps, return
code, signal, timeout, and tool versions. Preserve raw stdout/stderr and tool
output for failures as well as passes.

The recommended machine-local bundle is:

```text
artifacts/<run-id>/
├── README.md                 # question, command, and disposition pointer
├── manifest.json             # schema version, revisions, matrix, controls
├── environment.txt           # host/toolchain/GPU/library identity
├── cases.jsonl               # one append-only record per attempted case
├── cases/                    # per-case stdout, stderr, structured output
├── traces/                   # only the smallest reproducible failures
├── reports/                  # generated summaries, never hand-edited facts
└── SHA256SUMS                # hashes for files retained for review
```

Large traces, databases, and machine-local outputs stay under `artifacts/`.
When a result must reproduce from a fresh checkout, retain a small immutable
bundle under `docs/evidence/<run-id>/` containing a summary, provenance,
selected raw facts, and checksums. Follow
[`docs/evidence/README.md`](../../evidence/README.md); evidence is not plan
status.

### 5. Apply stop gates

Stop immediately on a GPU fault, signal, timeout, corrupted output, or failed
known-good control. Run a health check, rerun only the known-good control once,
and stop the campaign if it no longer passes. Never reset a GPU, kill another
user's process, or continue contaminated collection without authorization.

For RCCL or other process-fatal failures distinguish three layers:

1. a raw library reproducer with no BigCherry or llama.cpp integration;
2. the existing llama.cpp reduction probe and correctness policy;
3. a real production workload and profiling trace.

Do not treat a BigCherry guard that prevents a library call as evidence that
the library succeeded or failed. Do not weaken a safety guard to make an
exploratory failure reachable.

### 6. Triage and conclude

Use a closed result set defined by the brief, such as `pass`, `wrong_result`,
`init_failure`, `launch_failure`, `gpu_fault`, `signal`, `timeout`, and
`harness_failure`. Keep “not tested”, “inconclusive”, and “unsupported by
design” distinct from failure.

The final summary must state what was tested, which controls passed, whether
the hypothesis was supported/falsified/unresolved, the smallest reproducible
case and raw evidence, whether any performance claim is justified, known
confounders, and the disposition: promote, repeat with a changed brief,
archive, or discard.

### 7. Promote only after review

Create a formal plan item only when the result identifies a concrete, bounded
next action. Link the run-id and evidence bundle from the plan item. State
which exploratory assumptions were verified and which remain open.

If implementation would add or move `tools/bigcherry` code, read
[`tooling/TOOLING.md`](../tooling/TOOLING.md) first and confirm the owning
domain; do not create a second campaign, patch, or evidence framework.

Promotion into a formal experiment contract requires immutable source linkage,
declared controls, correctness criteria, acceptance thresholds, and provenance
from [`EXPERIMENT_CONTRACT.md`](EXPERIMENT_CONTRACT.md). A report summarizes
evidence; it must not manufacture measurements.

### 8. Archive without losing the decision

Keep a compact conclusion and provenance in `docs/evidence/<run-id>/` when it
is reusable. Retain large bundles under `artifacts/<run-id>/` according to the
retention policy. Superseded narrative belongs under `docs/archive/`, with links
to evidence and the plan item. Do not copy stale
status into a maintained reference page.

## RCCL heterogeneous qualification example

For the RCCL primer that motivated this workflow:

1. verify the exact RCCL library, code-object coverage, and runtime linkage;
2. run one fresh-process homogeneous control at an exact production size;
3. run one fresh-process heterogeneous raw-RCCL probe at the same size;
4. run the existing BigCherry reduction probe with explicit META/AUTO to
   verify guarded production behavior;
5. trace only the smallest reproducible raw failure;
6. expand to additional sizes or algorithm/protocol combinations only if the
   result is genuinely unresolved.

Current project evidence establishes that heterogeneous RCCL groups fail on
the tested stack and that algorithm/protocol overrides do not make them safe.
This primer is therefore not a search for a heterogeneous RCCL winner. Future
RCCL tuning qualification belongs to the same-architecture performance scope
in HI88; heterogeneous correctness work should use the validated META path.
See [`MULTI_GPU_DISPATCH.md`](../architecture/MULTI_GPU_DISPATCH.md), HI84,
HI85, and HI88 for current boundaries.

## Handoff checklist

- [ ] qualification brief and run-id are present;
- [ ] local dry-run tests pass;
- [ ] controls, exact sizes, and stop gates are recorded;
- [ ] environment, linkage, revisions, and raw outputs are preserved;
- [ ] every attempted case has one machine-readable result;
- [ ] no unsupported performance claim is made;
- [ ] reusable evidence is compacted and checksummed;
- [ ] disposition is explicit;
- [ ] follow-on plan items link the run and remaining uncertainty;
- [ ] substantive repository changes are recorded in the release ledger.
