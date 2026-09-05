# Testing reference map

This directory separates deterministic repository checks, hardware-free patch
evidence checks, hardware diagnostics, and final contract qualification. Read
the authority for the decision you are making; a passing diagnostic is not a
promotion verdict.

## Which document answers which question?

| Document | Authority class | Use it for |
| --- | --- | --- |
| [TEST.md](TEST.md) | Operational procedure | Repository gates, scoped dry-runs, correctness/timing checks, profiling, and the Brutus recipe |
| [PATCH_VALIDATION.md](PATCH_VALIDATION.md) | Normative patch lifecycle/evidence contract | Package requirements, named checks, evidence identity, status semantics, promotion, demotion, and re-promotion |
| [MULTI_GPU_LARGE_MODEL_VALIDATION.md](MULTI_GPU_LARGE_MODEL_VALIDATION.md) | Scoped empirical implementation note | Brutus observations and constraints for the tested large-model multi-GPU configuration |
| [RCCL_HETEROGENEOUS_RUNBOOK.md](RCCL_HETEROGENEOUS_RUNBOOK.md) | Hardware qualification runbook | RCCL source viability, crash isolation, topology identity, correctness, and admissible performance |
| [../../archive/COVERAGE_AUDIT.md](../../archive/COVERAGE_AUDIT.md) | Historical snapshot outside the maintained testing corpus | Reviewing the cited HI34 tuner-surface audit; verify current code before acting |

## Agent decision flow

1. Pin the source commit, patch composition, bound contract IDs, target
   architecture, hardware, and model identity.
2. Run the hardware-free repository gate:
   `PYTHONPATH=tools python -m bigcherry check` and the relevant test suite.
3. Inspect the canonical patch package and every bound Experiment Contract.
   Use `patch-lint`, `patch-validate`, and an explicitly scoped dry-run for
   application mechanics; never use an unscoped apply as a generic test.
4. Classify the next run as diagnostic or final contract qualification. The
   current final campaign paths are RD08 and RD73; RD04 benchmark and RD58
   state-restore modes are diagnostic-only.
5. Preserve stock/control/subject identity and run correctness/activation
   gates before interpreting performance. Keep diagnostic PASS, BLOCKED,
   FAIL, ERROR, and `not_applicable` distinct.
6. Persist the exact commands, environment, source/build/contract identity,
   measurements, correctness result, failures, and artifact references through
   the canonical evidence writer.
7. Promote only when complete current evidence exists for every bound
   contract and `eligible_for_validated_state` is true. On failure or a
   blocked run, preserve the evidence and reason; re-promotion requires fresh
   evidence at the current pin.

## Hardware and evidence boundary

Hardware-free checks establish package/repository mechanics and can expose
missing or malformed evidence. They cannot prove device correctness,
performance, topology safety, or an Experiment Contract threshold. Missing
hardware is `BLOCKED`; infrastructure or runner faults are `ERROR`; an
observed correctness or acceptance violation is `FAIL`. Only a canonical
contract qualification path may create the evidence used for a promotion
decision.

