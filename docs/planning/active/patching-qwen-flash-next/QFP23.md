---
id: QFP23
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-04T22:01:19.661087+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Runtime profiles from config files (no model-specific values in patch code) + 0910 loader rework

## 2026-10-12 authoritative rebaseline: implemented 0910, residual correctness gates

**State:** 0910 is already implemented and marked validated. Its shipped `engines/llamacpp/overlay/profile/*.ini`, C loader, Python validator and unit tests supersede the original implementation TODOs below. QFP23 remains pending only for the following correctness work. QFP18 owns historical stack-promotion evidence. No additional loader, profile selector, scheduler or config registry.

**Source-proven conditional gaps (not observed serving failures):**

1. `0910_feature_sets/patch.py::_N::bc_parse_profile_file` (lines 159–231) accepts duplicate identical `NAME=VALUE`, repeated `@include`, repeated `description`/`arch` and invalid `arch` syntax that `tools/bigcherry/patch/runtime_profiles.py::parse_text` rejects. `bc_flatten` rejects conflicting values, not identical duplicates. Runtime also omits the Python `documented_flags`/ENV_DOCS allowlist, so custom `BIGCHERRY_PROFILES` overrides can apply undocumented flags.
2. `bc_load_profiles` enumerates files without sorting; `ggml_bigcherry_features_apply_arch` selects the first matching `arch`. Python `check` permits two profiles claiming one architecture. Selection then depends on directory order. Shipped `flashnext` currently has the only `qwen4exp` claim.
3. `src/llama-model.cpp::llama_model_create` discards the `ggml_bigcherry_features_apply_arch` return; the server checks only earlier `ggml_bigcherry_features_init`. Failed auto application can therefore continue into model creation without requested settings. Preserve library no-exit, but propagate failure through the existing model-load error path.
4. `bc_auto_pending` is consumed on the first model and applies process-global environment values. Auto cannot provide per-model reconfiguration in a shared process, and flags read before model load cannot be retroactively set.

**Bounded patch/test contract:** Extend existing `test_runtime_profiles.py` and `test_0910_feature_sets.py` with identical Python/C fixtures for duplicate flags/includes/description/arch, malformed arch, undocumented flags, cycles, conflicts, overlong lines and rollback. Reject duplicate architecture ownership across files before mutation. In 0910 `_N_MODEL`, check late auto status and reject the model via its existing loader error path, never exit/abort. Preserve explicit-env precedence, named-profile behaviour, help=0 and release bundle identity. Verify early-read flags and restrict auto to first target per process; use separate model child processes for different settings. Run patch-lint, host C compile/tests and Linux/Windows startup smoke before qualification. No GPU sweep is justified for this parser contract.

**Upstream comparator:** `ggml-org/llama.cpp` `tools/server/server-models.cpp` already loads per-model INI presets and cascades options into child process arguments; `tools/server/README.md` documents precedence. Reuse that process isolation instead of adding global-env switching. https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-models.cpp

**Evidence and terminal gate:** Last independent 0910 source relocation 2026-10-09 04:20Z; no independent QFP23 implementation/PR/queued test in preceding 12 hours. Recent Radiance prefill/GDN/MoE/AllReduce/n-gram, QFP15 and release work is protected. Twelve source-static assertions and eight disposable host-model cases passed; **no** compiled-C repository test, HIP build or GPU benchmark was run. ACCEPT only with zero Python/C acceptance divergence, unique deterministic arch selection, no successful model load after failed auto application and unchanged explicit/named profile output. Otherwise restrict auto and retain named profiles. QFP18's historical +29%/+45% native-vs-stack decode is not a 0910 performance measurement.

## Historical implementation proposal (superseded where conflicting)

## Description

Owner 2026-10-05: model-specific configuration must not be hard-coded in patch code; it belongs in configuration files holding preconfigured settings per model, unless impossible. 0910 keeps only the mechanism (load, validate, apply); profiles (hip-q81, sched-async, flashnext, future qwen27b, ...) move to config/runtime-profiles/*.ini. Folds in deep-dive req_77d168caab954021 (0910 init hooks, no exit in library, whole-request validation + rollback, span parsing, ggml_bigcherry_getenv wrapper).

## Historical steps (substantially implemented; see current gate above)

1. Profile file format (INI-like, C-parseable): [name] sections; optional arch = <gguf arch>; description = ...; NAME = VALUE lines; @other includes. Default location next to the binary / install share dir, override BIGCHERRY_PROFILES=<file|dir>.
2. Python validator (bigcherry profiles check, wired into patch-lint/check): duplicates, unknown @refs, cycles, conflicting assignments, flag names must exist in some patch ENV_DOCS, value syntax.
3. 0910 C loader: read file(s) once in ggml_bigcherry_features_init(); flatten; validate the whole request before mutating env; rollback on failure; no exit/abort (help/list returns a status the tools map to exit codes); log explicit values as <explicit>; max sizes; no strtok.
4. Init hooks (deep-dive 1): ggml_init first line, get_reg() before the static registry, ggml_backend_load_all_from_path() before dlopen/score, GCC/Clang ctor fast path; tools call init at main and map HELP/ERROR.
5. ggml_bigcherry_getenv() wrapper; migrate BIGCHERRY_*/GGML_HIP_* consumers (inside each patch) so read order no longer matters.
6. BIGCHERRY_FEATURES=auto: after model load pick the profile whose arch matches; ENV_DOCS records when each flag is read (load-time vs lazy) so auto only claims lazily read flags.
7. Move per-model tuned values (1301 width, 1295 threshold, DFlash n_max, ATTN_TS / FFN_TS / DRAFT_VOCAB_N) into profiles; install the profiles dir with release builds.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline: validator rejects bad files; C loader unit tests (file parse, flatten, conflicts, rollback, help status, explicit-wins, overlong); MSVC compile. Hardware: release build with BIGCHERRY_FEATURES=flashnext from the file reproduces v6; auto picks flashnext on Flash-Next and the 27B profile on the 27B.

## Effort & Risk



## Standards



## Acceptance Criteria

- No model-specific values compiled into patch code except intrinsic model code paths (1303 arch check, 1308/1327 inside qwen4exp.cpp).
- Profiles live in config files, validated by tooling and by the loader.
- Deep-dive 1 findings resolved.

## Notes

What stays in code: behaviour that is part of one model's own code path (qwen4exp.cpp patches, 1303 refusing non-qwen4exp) - code semantics, not tunables.

Owner 2026-10-05 (two directions): (1) profile files must live in a folder relative to the final release - we still ship a releasable package; (2) incorporate per-model tuning. Release layout: canonical source in the overlay src/bigcherry-profiles/ (part of source identity), CMake copies it next to the binaries as bin/bigcherry-profiles.ini (+ install rule), resolve_runtime_artifacts includes it in the runtime bundle hash, the C loader finds it relative to libggml-base (dladdr / GetModuleHandleEx) with BIGCHERRY_PROFILES=<file> override. Per-model tuning: model profiles ([flashnext], [qwen27b], ...) with arch = <gguf arch>; BIGCHERRY_FEATURES=auto picks by architecture after model load (only lazily-read flags; ENV_DOCS records read time); a per-model sweep harness (tools/lab/tune-profile) sweeps each patch's knobs per model (1301 MAXCOLS, 1295 GATHER_MIN, DFlash n_max/placement, AR thresholds, QSA_CHUNK, ATTN_TS/FFN_TS) and writes the winners into that model's profile section with the evidence run id in a comment. First data: 1301 width sweep + DFlash sweep -> [qwen27b].

## Change Log

- 2026-10-04T22:01:19.661087+00:00 (created-by): Created by agent
- 2026-10-04T22:04:04.431702+00:00 (updated-by): Updated: section:notes

## Ledger-events



- chg_20261004_235327_model-and-scope-tuning-now-liv_4754
- 2026-10-04T23:53:33.547468+00:00 (updated-by): Updated: section:ledger-events
- chg_20261004_235343_model-and-scope-tuning-now-liv_4846
- 2026-10-04T23:53:49.845549+00:00 (updated-by): Updated: section:ledger-events
- chg_20261005_113349_the-fused-decode-kernels-now-a_9568
- 2026-10-05T11:33:52.589848+00:00 (updated-by): Updated: section:ledger-events
