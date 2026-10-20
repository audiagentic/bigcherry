# MMQ candidate catalog and architecture targeting

This is the detail behind `bigcherry --engine llamacpp generate --arch ...`
moved out of the top-level README (which stays orientation-only).

## Targets

Every AMD GPU llama.cpp supports — 26 architectures from GCN4 through RDNA4.
`--arch` accepts individual targets or group names (`all`, `rdna`, `rdna3`,
`rdna3.5`, `rdna4`, `cdna`, `gcn`).

Targeting everything does not multiply the catalog, because MMQ candidates are
keyed by their *resolved config*: the many architectures sharing a config table
share one candidate carrying all their architecture bits. All 26 targets
produce 3062 candidates against 2732 for two.

The architecture enumeration lives in `tools/bigcherry/tuning/schema.py` and
is **append-only** — each entry's index is its bit position in the persisted
`architecture_mask`. The C++ enum is generated from it into
`hip-autotune-arch.h`, so the two languages cannot drift.

## MMQ config tables are sparse

Upstream's MMQ `CASE` tables do not define every (type, J, fallback)
combination — CDNA defines 154 rows where RDNA3 defines 260, and `q8_0` on CDNA
stops at J=64. `ggml_cuda_mmq_get_config` returns a sentinel for undefined
combinations and the native search skips them.

The catalog is therefore **derived from the tables**, never from
`range(8, 129, 8)`. A row upstream adds becomes a candidate on the next
`generate` with nothing to remember to update; `artifacts/<rev>/
mmq-table-coverage.json` records what each table defined, so diffing two
releases shows exactly which solutions came or went.

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

## Repo layout reference

| Path | Contents |
| --- | --- |
| `src/` | New files, mirroring the llama.cpp tree at their final paths |
| `engines/llamacpp/patches/` | Anchored edits to upstream-owned files |
| `tools/bigcherry/` | The `bigcherry` Python CLI |
| `tools/lab/` | Temporary plan-owned investigation; no copied upstream source or raw run output |
| `tools/bigcherry/source/audit.py` | Strict-mode upstream invariant audit (HI01) |
| `tools/bigcherry/tuning/catalog.py` | Candidate catalog generator — single source of truth (HI03) |
| `tools/bigcherry/tuning/schema.py` | Candidate manifest JSON schema (HI03) |
| `sql/` | Current SQLite schema plus tested migration history for record/tune modes; runtime databases are not tracked |
| `vendor/llama.cpp/` | The checkout we patch and build (not tracked) |
| `artifacts/` | Audit JSON, manifests, exported caches (not tracked) |
| `docs/reference/` | Maintained cross-cutting guidance and runbooks |
| `docs/standards/` | Project standards — normative |
| `docs/planning/` | Active/completed plan items, design decisions, and work history |
| `docs/evidence/` | Compact tracked evidence retained for reproducible decisions |
| `releases/` | Machine-readable per-pin compatibility/rebase state consumed by tooling |
| `docs/releases/` | Human-facing release notes, audit/check-in guidance, and release ledger |

`vendor/llama.cpp` is a real working tree, not a scratch copy: builds run from
it in place. See `sql/README.md` for dispatch-database schema ownership.

Repository scratch is never source or evidence. Put one-off/generated outputs
under ignored `artifacts/`, `work/`, or a host temporary directory; root
`_scratch*`/`scratch` paths and runtime SQLite databases are intentionally
untracked.
