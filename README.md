# BigCherry

BigCherry develops and qualifies improvements to upstream inference engines, with
a focus on AMD GPUs. Changes live in small, reviewable packages alongside
upstream source rather than in a permanent fork. Performance and correctness
claims are tied to measurements, not just successful builds.

## What BigCherry adds

- **Measured GPU dispatch for llama.cpp.** Record the shapes a workload uses,
  measure eligible HIP kernels, then replay selected implementations without
  benchmarking during serving. A cache miss leaves upstream dispatch in charge.
- **Targeted engine patches.** Package upstream-owned changes with explicit
  lifecycle state, dependencies, checks and evidence. Reapply them against a
  pinned engine revision, or leave rejected experiments out of production.
- **Radiance kernel extensions.** Build separate AMD kernel libraries and
  remote-expert workers against the pinned Radiance engine. Engine patches
  provide narrow integration points; component code stays under BigCherry's
  control.
- **Reproducible qualification.** Share one build, benchmark, quality-gate and
  release workflow across both engines. Only measured, reviewed changes move
  into production sources.

## Engines

| Engine | BigCherry's work | Details |
| --- | --- | --- |
| **llama.cpp** | HIP autotuning, dispatch, and a catalog of upstream patches | [Overview](docs/reference/engines/llamacpp/README.md) · [Patch catalog](docs/reference/engines/llamacpp/PATCHES.md) |
| **Radiance** | RDNA kernel components, remote MoE execution and focused engine patches | [Overview](docs/reference/engines/radiance/README.md) · [Extension catalog](docs/reference/engines/radiance/ENHANCEMENTS.md) |

## Find your way around

- [Getting started](GETTING_STARTED.md) — local setup, builds, tests and
  validation commands.
- [Reference](docs/reference/README.md) — architecture, engines, patching,
  testing and tooling.
- [Patch and component documentation standard](docs/reference/DOC_STRUCTURE.md)
  — one owning page per change, with summary, state and detail.
- [Planning](docs/planning/) — active designs and work in progress;
  [release notes](docs/releases/) and [evidence](docs/evidence/) hold the
  corresponding records.

The root README is an orientation, not a configuration manual or benchmark
leaderboard. Engine-specific behavior and evidence belong on the linked pages.
