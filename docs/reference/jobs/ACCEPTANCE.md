# Job-service cutover acceptance

RCD10 retires the lab queue only after software gates are green and the current accepted hardware/production environment passes real acceptance. Never simulate an absent capability as accepted.

## 1. Generate the capability matrix

Create canonical BatchSpecs for every active qualification variation, then run:

```bash
python -m bigcherry jobs acceptance specs/*.json
```

The result is `bigcherry.jobs.acceptance-matrix.v1`. Cases are keyed by scientific/resource capability: executor/platform, patch, architecture, model, producer, toolchain path, GPU count/VRAM/peer requirement, production-lane flag and common-patch composition. Physical GPU slot/index is not part of a case.

`ready=true` means every requested case can bind against current **accepted** inventory. Missing architecture/cards, ambiguous same-architecture models, insufficient VRAM/peer topology, missing executor inventory, or target host/platform mismatch remain visible as `supported=false` with a reason. Exit status is 2 while any case is unsupported.

## 2. Required software gates

Before Brutus cutover:

```bash
python -m unittest discover -s tools/tests/jobs -p 'test_*.py' -v
python -m bigcherry jobs executors doctor
python -m bigcherry jobs status
```

GitHub `Jobs service validation` must be green. Real Noble Slurm CI is a scheduler/service reference only; it does not replace Brutus GPU acceptance.

## 3. Brutus hardware gates

Record exact accepted inventory hash and run:

```bash
python -m bigcherry hardware discover brutus --record
python -m bigcherry hardware diff brutus
python tools/admin/render_bigcherry_slurm.py \
  --hardware-root <work>/hardware --executor-id brutus \
  --node-name brutus --cpus <N> --real-memory-mib <MiB>
sudo slurmd -G
```

For each supported allocation shape/toolchain, prove stable-ID allocation attestation, HIP initialization/properties, peer/tensor split where required, llama-bench/server smoke and required producer preflights. Keep `ConstrainDevices=no` unless the complete ROCm device-cgroup matrix succeeds.

## 4. Production gates

For timed measurements prove both dynamic paths:

- target intersects current/potential llama-swap claim -> exclusive production window;
- target is disjoint -> no co-resident gating evidence until the loaded-idle isolation experiment qualifies it.

Ambiguous/malformed production ownership fails closed to exclusive. Production config/process drift or unexpected target-GPU use invalidates the sample as environment contamination.

## 5. Incident matrix

Force and record at least:

- harness failure + retry as new attempt;
- client disconnect after durable submit;
- hold/release/cancel;
- submission-intent crash recovery with no duplicate native execution;
- branch/scientific-identity drift;
- hardware inventory drift;
- dirty/staged evidence destination refusal;
- controller restart with Slurm ownership retained;
- production contamination/window recovery;
- disk/resource preflight failure.

Scientific FAIL is a normal completed experiment and must not be converted into a harness retry.

## 6. Full-series gate

Complete at least one predeclared full series with no shell watcher intervention:

```text
submit -> ingest/execute -> terminal attempts -> harvest --commit -> report
```

Require:

- planned N unchanged;
- exact frozen scientific identity and hardware cohort throughout;
- committed verified evidence;
- report statistics/verdict semantics matching direct campaign evidence;
- no manual evidence copy/stash/implicit git staging.

## 7. Soak and retirement

Migrate pending lab entries as **new** JobSpecs; never import old `CAMPAIGN_EXIT=` lines as completed managed attempts. Disable old queue submission during soak but keep rollback documented. Retire/archive `queue.sh`, `run_campaign.sh`, `make-serial-2.sh` and hand-written watcher/switch scripts only after a full planned series plus representative alternate capability cases complete under the managed path.

Rollback: pause new BigCherry submissions, stop managed scheduler execution before using the direct/manual campaign path, and preserve the run store read-only. Never operate two schedulers against the same GPUs concurrently.
