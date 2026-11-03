# ServingManifest v1 — errors and exits

Status: **locked proposed convention**, 2026-11-02. This document describes one error interface for `bc-serve`, the managed Radiance engine adapter, `r3_moe_worker`, and config-sensitive plugin setup. It does not introduce a replacement for Radiance's status codes.

## 1. Typed C++ error

```cpp
enum class ConfigErrorCode : uint16_t {
    unknown_field,
    missing_field,
    duplicate_field,
    invalid_type,
    invalid_value,
    unsupported_version,
    invalid_format,
    digest_mismatch,
    experiment_forbidden,
    unresolved_reference,
    provenance_invalid,
    calibration_invalid,
    ownership_mismatch,
    plan_mismatch,
    admission_refused,
    io_failure,
    abi_mismatch,
    invalid_state
};

struct ConfigError {
    ConfigErrorCode code;
    int status;                        // derived by status_of(code): RAD_E_*
    std::string path;                  // dotted/indexed human field path, or "" for whole file
    std::string message;               // stable plain-English diagnostic
    std::string hint;                  // actionable fix; empty if none
    std::optional<std::string> source_file;  // TOML/JSON/source path; not raw JSON
    std::optional<uint32_t> line;      // TOML/JSON line if available
    std::optional<uint32_t> column;    // TOML/JSON column if available
};

template<class T>
using ConfigResult = std::expected<T, ConfigError>; // C++23 spelling is illustrative;
                                                     // C++20 implementation uses a small variant.
```

`status` is computed from the enum and MUST NOT be supplied independently by call sites. Error messages may include values and details but must never silently change the code. Source-file locations come from the input parser where available; runtime/lower-level validations may have null source locations. No C++ exceptions cross a C plugin ABI.

## 2. Stable public error codes

The machine `code` field uses the following stable spelling. Every code maps to exactly one upstream `RAD_E_*` numeric category from `abi/rad_types.h` at upstream commit `4a61d72`.

| Machine code | RAD status | Numeric | Meaning |
|---|---|---:|---|
| `BC_CFG_UNKNOWN_FIELD` | `RAD_E_SCHEMA` | -9 | Unsupported input or manifest field, including unsupported curated engine option |
| `BC_CFG_MISSING_FIELD` | `RAD_E_SCHEMA` | -9 | Required field or entire section absent |
| `BC_CFG_DUPLICATE_FIELD` | `RAD_E_DUPLICATE` | -11 | Duplicate JSON member, role, GPU BDF, calibration match or repeated entry |
| `BC_CFG_INVALID_TYPE` | `RAD_E_INVAL` | -12 | JSON/TOML type not valid for the field |
| `BC_CFG_INVALID_VALUE` | `RAD_E_INVAL` | -12 | Range, enum, unit or cross-field constraint violation |
| `BC_CFG_UNSUPPORTED_VERSION` | `RAD_E_UNSUPPORTED` | -18 | Unsupported `apiVersion`, `kind`, contract version or plan version |
| `BC_CFG_INVALID_FORMAT` | `RAD_E_FORMAT` | -14 | Invalid UTF-8, malformed JSON, nonfinite or non-I-JSON number, or malformed immutable document |
| `BC_CFG_DIGEST_MISMATCH` | `RAD_E_FORMAT` | -14 | Missing, malformed or incorrect JCS digest or owner-set digest encoding |
| `BC_CFG_EXPERIMENT_FORBIDDEN` | `RAD_E_SCHEMA` | -9 | Any experimental field in a production profile, even a false/default-valued one |
| `BC_CFG_UNRESOLVED_REFERENCE` | `RAD_E_NOTFOUND` | -17 | Unrecognized helper role, absent referenced GPU or required model/calibration identity |
| `BC_CFG_PROVENANCE_INVALID` | `RAD_E_SCHEMA` | -9 | Bad or missing effective-source coverage, pointer, dependency or cycle |
| `BC_CFG_CALIBRATION_INVALID` | `RAD_E_SCHEMA` | -9 | Ambiguous, missing, unsupported or unevidenced calibration selection |
| `BC_CFG_OWNERSHIP_MISMATCH` | `RAD_E_SCHEMA` | -9 | Initial owner map, per-helper set, count, device bytes or digest inconsistent |
| `BC_CFG_PLAN_MISMATCH` | `RAD_E_SCHEMA` | -9 | Stable expected and live engine-plan consumers do not agree |
| `BC_CFG_ADMISSION_REFUSED` | `RAD_E_NOMEM` | -6 | Resources cannot admit the required plan, including insufficient live claimable VRAM |
| `BC_CFG_IO_FAILURE` | `RAD_E_IO` | -13 | Manifest/cache/measurement read or write failure |
| `BC_CFG_ABI_MISMATCH` | `RAD_E_ABI` | -10 | Plugin config struct size/version or engine binding ABI mismatch |
| `BC_CFG_INVALID_STATE` | `RAD_E_STATE` | -16 | Wrong configuration-installation phase, repeated immutable install or startup state corruption |

The public error-code spelling is part of the v1 contract. No independent `BC_E_*` numeric namespace. Local C++ enum numbers are implementation-only; `status_of(code)` maps to Radiance's stable integers. If an error is impossible to classify, return `BC_CFG_INVALID_STATE`, not an untyped exception or a silent fallback. Plugin APIs return the mapped `RAD_E_*` integer and can provide a diagnostic through an explicitly versioned API or a top-level adapter; never throw across the C ABI.

The reader checks the complete manifest and fails closed. A correctly rehashed document with invalid ownership or memory semantics still fails with its *semantic* code. A stale manifest with both invalid hash and invalid semantics first fails with `BC_CFG_DIGEST_MISMATCH`.

## 3. JSON error shape

The JSON renderer emits exactly these keys; nullable location fields are explicit nulls. Optional `instance_id` and `role` are added only when known. It never includes a successful result in the same object.

```json
{
  "code": "BC_CFG_EXPERIMENT_FORBIDDEN",
  "status": -9,
  "path": "spec.experiment",
  "message": "spec.experiment is forbidden when spec.intent.service.profile=production",
  "hint": "select profile=lab explicitly and regenerate the manifest",
  "source_file": "site.toml",
  "line": 27,
  "column": 1
}
```

The full error is printed on a single JSON line when `bc-serve --json-errors` is selected, to support logs and automation. The convention resembles RFC 9457 structured problem details but is **not** an HTTP `application/problem+json` response: there is no invented HTTP `status` or `type` URL. If the service later exports HTTP problems, add a separate RFC 9457 adapter.

`path` is the **human dotted/indexed path**, e.g. `spec.processes.helpers[2].vram_budget_mib`; the manifest's provenance uses strict JSON Pointers (e.g. `/spec/processes/helpers/2/vram_budget_mib`). Never silently mix the two syntaxes. Errors in decoding the entire document use `path=""`.

## 4. Human rendering

For the JSON example above:

```text
bc-serve: config: BC_CFG_EXPERIMENT_FORBIDDEN (RAD_E_SCHEMA, -9)
  spec.experiment: spec.experiment is forbidden when spec.intent.service.profile=production
  source: site.toml:27:1
  hint: select profile=lab explicitly and regenerate the manifest
```

For a memory admission failure:

```text
radiance: config: BC_CFG_ADMISSION_REFUSED (RAD_E_NOMEM, -6)
  spec.plan.engine_memory.expected.vram.kv_resolved: admission: resolved KV: live 18906873856 bytes fall short of the required 18907078656 bytes (kv_required 18907078656, the common KV budget after the staging split and MiB rounding)
  hint: free the engine card, or choose a smaller service intent, and resolve a new manifest
```

The byte amounts are the shared vector `prod-kv-truncation-below` (`admission-vectors.json`): the production example's geometry at a claimable reading where the unrounded KV share still covers `kv_required` but the whole-MiB cache the engine allocates does not. They are generated by the admission model, not a measurement. This block is exactly what the shared reader's `render_human` writes for that vector; `bc_v1_test` checks it.

Every process uses its own executable prefix (`bc-serve`, `radiance`, `r3_moe_worker`) followed by `: config:`. The shared library **never prints** an error while returning it. An adapter must not print and then let a caller print the same failure again. No secrets, entire contents of a calibration file, or large manifest fragments appear in error messages.

## 5. Process exits

| Condition | bc-serve | radiance | r3_moe_worker |
|---|---:|---:|---:|
| Successful startup or successful dry-run/validate/plan-only operation | 0 | 0 | 0 |
| Invalid config, unsupported version, format/hash, role, experiment, semantic mismatch or ABI mismatch | **2** | **2** | **2** |
| Manifest/model/cache file access or atomic write failure | **3** | **3** | **3** |
| Insufficient host/VRAM/pinned/cgroup resources or device admission failure | **4** | **4** | **4** |
| Internal state violation, unexpected loader/plugin state or unexpected implementation failure | **5** | **5** | **5** |

The CLI translates negative `RAD_E_*` statuses to these **positive process exits**, never returns the negative status directly from `main()`. Any fatal signal follows normal process signal semantics; the supervisor reports the original signal and stops the serving set. The supervisor preserves each child's diagnostic and records the failing role/digest without substituting a generic error. Partial launch is torn down; no invalid manifest is retried with CLI flags or inherited settings.

Exit 4 is used for a real resource-admission refusal even if the operator could recover by changing resource settings. A missing required resource **field**, however, is exit 2. A missing readable input file is exit 3. All configuration errors must be detected before entering GPU execution where feasible.

## 6. Required error tests

- Unknown `engine.execution` key ⇒ `BC_CFG_UNKNOWN_FIELD` / -9 / exit 2.
- `profile=production` plus any `experiment` key, even `{ }` or all defaults ⇒ `BC_CFG_EXPERIMENT_FORBIDDEN` / -9 / exit 2.
- Bad hash ⇒ `BC_CFG_DIGEST_MISMATCH` / -14 / exit 2.
- Correctly rehashed invalid owner map ⇒ `BC_CFG_OWNERSHIP_MISMATCH` / -9 / exit 2.
- Correctly rehashed stable memory-plan drift (statics, expert floor or ceiling, KV ceiling or expert-plane geometry; a card capacity below the measured one; an automatic placement that resolves differently from the measured plan) ⇒ `BC_CFG_PLAN_MISMATCH` / -9 / exit 2. A live staging region or expert residency that differs from the measured one is not drift, and neither is a card capacity above the measured one.
- Each admission check 1 byte short (expert residency below its floor; claimable below statics + live experts + live stage + kv_required; resolved KV below kv_required) ⇒ `BC_CFG_ADMISSION_REFUSED` / -6 / exit 4; equality passes; a reading below the old measured split but at or above the required split passes. The shared vectors in `admission-vectors.json` are the cases every implementation must reproduce.
- Missing `helper2` or a nonexistent GPU BDF ⇒ `BC_CFG_UNRESOLVED_REFERENCE` / -17 / exit 2.
- Invalid `bc_r4d_configure_helper` struct version ⇒ `BC_CFG_ABI_MISMATCH` / -10 / exit 2.
- Missing manifest file ⇒ `BC_CFG_IO_FAILURE` / -13 / exit 3.
- Decode/validation errors must preserve code, path, source location and hint across all three entry points.

All renderer tests use the same `ConfigError` object and check exact machine codes, numerical statuses and exit classes.
