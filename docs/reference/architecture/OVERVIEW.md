# BigCherry architecture overview

BigCherry is a release-tolerant overlay around an upstream llama.cpp checkout.
The current base is identified by the active source and release records, not
by a revision embedded in this document. Use the relevant plan item for work
status and the release ledger for what has been published.

## Repository layers

- `vendor/llama.cpp/` is the upstream checkout used for audits and builds.
- `src/` contains complete files owned by BigCherry and overlaid at the
  corresponding upstream paths.
- `patches/<patch-id>/` contains anchored edits to upstream-owned files. Each
  package owns its metadata, validation, fixtures, tools, and evidence.
- `tools/bigcherry/` contains the maintained Python CLI and domain packages;
  `tools/tests/` contains permanent offline tests.
- `config/` defines sources, recipes, platforms, and experiment contracts.
- `docs/planning/` owns current work and decisions; `docs/evidence/` holds
  compact reproducible run evidence; `artifacts/` holds large or transient
  outputs.

## Execution flow

The normal lifecycle is:

1. resolve or fetch an upstream source and record its immutable identity;
2. audit the source shape and verify patch anchors;
3. apply the packaged overlay into the vendor checkout;
4. generate a candidate manifest from the patched tree;
5. build a named lane (native, record, tune, or replay);
6. measure eligible candidates on a real workload;
7. validate correctness and provenance before promotion;
8. export a compact replay cache for production.

Each stage is idempotent and later stages fail closed when their inputs are
missing, stale, or bound to a different source/build identity.

## Dispatch model

Runtime dispatch uses canonical operation signatures and a fixed set of kernel
families (`mmvq`, `mmq`, `mmvf`, `mmf`, and `blas`). Candidate catalogues are
derived from the source and configuration; they are not duplicated in plan
documents or model-specific code. Candidate identity, source provenance,
experiment identity, and workload metadata remain separate namespaces.

Native execution and forced-variant execution share the same launcher. The
record/tune path may measure candidates and write journals or databases; the
replay path only loads a verified compact cache and falls back to native when
an entry does not match the current identity.

## Multi-GPU and experiments

Tensor splitting and reduction-provider selection are separate concerns.
RCCL, internal, and metadata/fallback paths are selected under explicit
topology and capability checks; telemetry records what actually ran without
changing dispatch identity. See
[`MULTI_GPU_DISPATCH.md`](MULTI_GPU_DISPATCH.md) for the provider contract.

Experiment contracts in `config/experiment-contracts.toml` describe an
optimization's hypothesis, source, scope, positive/control workloads,
boundaries, correctness requirements, and acceptance thresholds. They expand
through the existing campaign and evidence machinery; they do not introduce
a second benchmark or dispatch system. See
[`../experiments/EXPERIMENT_CONTRACT.md`](../experiments/EXPERIMENT_CONTRACT.md).

## Where current truth lives

- implementation behavior: `src/`, `patches/`, and `tools/bigcherry/`;
- current work, acceptance, and decisions: `docs/planning/active/`;
- released change history: `docs/releases/` and the release ledger;
- reproducible run evidence: `docs/evidence/`;
- historical snapshots and handovers: `docs/archive/`.

This page is intentionally a stable architecture map, not a phase tracker or
historical handoff.
