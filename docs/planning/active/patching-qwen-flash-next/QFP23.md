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

## Description

Owner 2026-10-05: model-specific configuration must not be hard-coded in patch code; it belongs in configuration files holding preconfigured settings per model, unless impossible. 0910 keeps only the mechanism (load, validate, apply); profiles (hip-q81, sched-async, flashnext, future qwen27b, ...) move to config/runtime-profiles/*.ini. Folds in deep-dive req_77d168caab954021 (0910 init hooks, no exit in library, whole-request validation + rollback, span parsing, ggml_bigcherry_getenv wrapper).

## Steps

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

## Change Log

- 2026-10-04T22:01:19.661087+00:00 (created-by): Created by agent
