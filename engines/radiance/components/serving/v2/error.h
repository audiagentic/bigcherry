#pragma once
/* The one configuration error of the serving manifest v2 (docs/design/serving-manifest-v1/errors.md §1-§5): a stable
 * code, the Radiance status it maps to (computed, never supplied), a human dotted/indexed path, a stable message, an
 * actionable hint and an optional source location. The library returns errors and never prints them. */
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace bc::serving::v2 {

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
    ConfigErrorCode code = ConfigErrorCode::invalid_state;
    int status = 0;                           // status_of(code); set by make_error, never by a call site
    std::string path;                         // "spec.processes.helpers[2].vram_budget_mib", or "" for the document
    std::string message;
    std::string hint;
    std::optional<std::string> source_file;
    std::optional<uint32_t> line;
    std::optional<uint32_t> column;
};

/* The only way to build an error: the status follows from the code. */
ConfigError make_error(ConfigErrorCode code, std::string path, std::string message, std::string hint = "");

int status_of(ConfigErrorCode code);                  // RAD_E_* (abi/rad_types.h)
std::string_view machine_name(ConfigErrorCode code);  // "BC_CFG_UNKNOWN_FIELD"
std::string_view status_name(int status);             // "RAD_E_SCHEMA"
/* The process exit for an error (errors.md §5): 2 invalid configuration, 3 file access, 4 resource admission,
 * 5 internal state. */
int exit_code_of(const ConfigError& e);

/* errors.md §4: "<prefix>: config: CODE (RAD_E_X, -n)\n  path: message\n  source: f:l:c\n  hint: ...\n". */
std::string render_human(std::string_view executable, const ConfigError& e);
/* errors.md §3: one JSON object on one line; nullable location fields are explicit nulls. */
std::string render_json(const ConfigError& e);

/* A value or an error (C++20; errors.md's std::expected spelling is illustrative). */
template <class T>
class Result {
public:
    Result(T value) : v_(std::move(value)) {}
    Result(ConfigError error) : v_(std::move(error)) {}
    bool ok() const { return v_.index() == 0; }
    explicit operator bool() const { return ok(); }
    T& value() { return std::get<0>(v_); }
    const T& value() const { return std::get<0>(v_); }
    const ConfigError& error() const { return std::get<1>(v_); }

private:
    std::variant<T, ConfigError> v_;
};

/* Success, or the error: for checks that produce no value. */
using Status = std::optional<ConfigError>;

}  // namespace bc::serving::v2
