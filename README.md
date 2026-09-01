# bigcherry

Measured-dispatch autotuner for llama.cpp — packaged as a **release-tolerant
overlay** rather than a fork.

New to this repo? See `GETTING_STARTED.md` for build/test commands, how to
run a patch, and operational gotchas.

Upstream llama.cpp moves fast. Maintaining a long-lived fork means resolving
merge conflicts on every release. bigcherry instead keeps its work in two
buckets:

- **`src/`** — whole files that are *new* to the tree. The directory mirrors the
  llama.cpp layout exactly, so applying it is a copy and the eventual upstream
  PR diff falls straight out of it. New files never conflict.
- **`patches/`** — small, *anchored* edits to files upstream already owns. Each
  edit locates its insertion point by regex anchor and verifies the result, so
  it survives ordinary upstream churn and fails loudly — naming the anchor that
  went missing — when it does not.

The workflow for a new llama.cpp release is:

```bash
python -m bigcherry pull   --ref b1234        # fetch/checkout upstream
python -m bigcherry audit                     # strict invariant audit
python -m bigcherry apply --source bigcherry  # src/ overlay + anchored patches
python -m bigcherry generate --arch all       # candidate catalog -> artifacts
python -m bigcherry build   --profile replay-full
```

Every stage is idempotent, and every stage after `audit` refuses to run on a
tree that has not passed a strict audit for its current revision.

## Targets

Every AMD GPU llama.cpp supports — 26 architectures from GCN4 through RDNA4.
`--arch` accepts individual targets or group names (`all`, `rdna`, `rdna3`,
`rdna3.5`, `rdna4`, `cdna`, `gcn`).

Targeting everything does not multiply the catalog, because MMQ candidates are
keyed by their *resolved config*: the many architectures sharing a config table
share one candidate carrying all their architecture bits. All 26 targets
produce 3062 candidates against 2732 for two.

The architecture enumeration lives in `tools/bigcherry/autotune_schema.py` and
is **append-only** — each entry's index is its bit position in the persisted
`architecture_mask`. The C++ enum is generated from it into
`hip-autotune-arch.h`, so the two languages cannot drift.

### MMQ config tables are sparse

Upstream's MMQ `CASE` tables do not define every (type, J, fallback)
combination — CDNA defines 154 rows where RDNA3 defines 260, and `q8_0` on CDNA
stops at J=64. `ggml_cuda_mmq_get_config` returns a sentinel for undefined
combinations and the native search skips them.

The catalog is therefore **derived from the tables**, never from
`range(8, 129, 8)`. A row upstream adds becomes a candidate on the next
`generate` with nothing to remember to update; `artifacts/<rev>/
mmq-table-coverage.json` records what each table defined, so diffing two
releases shows exactly which solutions came or went.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | New files, mirroring the llama.cpp tree at their final paths |
| `patches/` | Anchored edits to upstream-owned files |
| `tools/bigcherry/` | The `bigcherry` Python CLI |
| `tools/bigcherry/source_audit.py` | Strict-mode upstream invariant audit (HI01) |
| `tools/bigcherry/autotune_catalog.py` | Candidate catalog generator — single source of truth (HI03) |
| `tools/bigcherry/autotune_schema.py` | Candidate manifest JSON schema (HI03) |
| `sql/dispatch-db.sql` | SQLite schema for record/tune modes |
| `vendor/llama.cpp/` | The checkout we patch and build (not tracked) |
| `artifacts/` | Audit JSON, manifests, exported caches (not tracked) |
| `docs/standards/` | Project standards — normative |
| `docs/planning/` | Work-item plan (HI01–HI16) |

`vendor/llama.cpp` is a real working tree, not a scratch copy: builds run from
it in place.

## Build profiles

See `docs/standards/HIP_AUTOTUNE_STANDARDS.md` §6.2.

| Profile | Candidates | Tuner | SQLite |
| --- | --- | --- | --- |
| `inventory` | native only | no | yes (record) |
| `workload-max` | from inventory JSON | yes | yes |
| `full-max` | all bounded matrix | yes | yes |
| `replay-full` | workload-max set | **no** | **no** |
| `replay-slim` | winners + native fallback | **no** | **no** |

## Runtime control

| Variable | Values |
| --- | --- |
| `GGML_HIP_DISPATCH_MODE` | `record`, `tune`, `replay` |
| `GGML_HIP_DISPATCH_DB` | path to sqlite file (record/tune) |
| `GGML_HIP_DISPATCH_CACHE` | path to compact replay cache (replay) |
| `GGML_HIP_DISPATCH_MISS` | `native`, `native-record` |
