# Reference — by concern

| Document | What it is | When to read it |
| --- | --- | --- |
| [HANDOFF.md](HANDOFF.md) | State of play + next actions | Picking work up; first stop |
| [OVERVIEW.md](OVERVIEW.md) | Phases, status, principles, verification | Understanding scope and progress |
| [FAMILY_MODEL.md](FAMILY_MODEL.md) | Six families, identity rules, rejected proposals | Designing a new family or candidate identity |
| [COVERAGE_AUDIT.md](COVERAGE_AUDIT.md) | What the tuner can/cannot choose between; gaps | Understanding why a signature has few options |
| [BUILD_AND_TEST.md](BUILD_AND_TEST.md) | Environment, cmake, test invocations, modes | Building, testing, or running on hardware |
| [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) | Architecture decisions + operational gotchas | Understanding why something is shaped a certain way; avoiding traps |
| [CANDIDATES.md](CANDIDATES.md) | Auto-generated candidate inventory | Looking up a specific candidate's config |
| [FINDINGS.md](FINDINGS.md) | Bugs and results notable outside bigcherry — kernel/llama.cpp/ROCm bugs, exceptional tunes | You just found something a stranger to this project would want to know about |
| [FINDINGS.md](FINDINGS.md) | Bugs and results notable outside bigcherry — kernel/llama.cpp/ROCm bugs, exceptional tunes | You just found something a stranger to this project would want to know about |
| [TUNING-DETAIL.md](TUNING-DETAIL.md) | Per-tune results (regenerated) | Reviewing hot signatures from a specific run — consult the DB for current numbers |
| [PACK_REVIEW.md](PACK_REVIEW.md) | Deltas vs. prework pack | Tracing why a plan diverged from the original spec |

**Tuning results and model-specific data live in the database**, not in these documents. Consult the tuning DB for per-signature winners, rankings, improvements, and coverage numbers.

**Archived originals** are in [archive/](archive/) — the previous structure before reorganization.
