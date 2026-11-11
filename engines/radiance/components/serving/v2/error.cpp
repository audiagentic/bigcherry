#include "error.h"

#include <rad_types.h>

#include <nlohmann/json.hpp>

namespace bc::serving::v2 {
namespace {
struct Row {
    std::string_view name;
    int status;
};

/* errors.md §2: every code maps to exactly one RAD_E_* category. */
Row row(ConfigErrorCode c) {
    switch (c) {
        case ConfigErrorCode::unknown_field: return {"BC_CFG_UNKNOWN_FIELD", RAD_E_SCHEMA};
        case ConfigErrorCode::missing_field: return {"BC_CFG_MISSING_FIELD", RAD_E_SCHEMA};
        case ConfigErrorCode::duplicate_field: return {"BC_CFG_DUPLICATE_FIELD", RAD_E_DUPLICATE};
        case ConfigErrorCode::invalid_type: return {"BC_CFG_INVALID_TYPE", RAD_E_INVAL};
        case ConfigErrorCode::invalid_value: return {"BC_CFG_INVALID_VALUE", RAD_E_INVAL};
        case ConfigErrorCode::unsupported_version: return {"BC_CFG_UNSUPPORTED_VERSION", RAD_E_UNSUPPORTED};
        case ConfigErrorCode::invalid_format: return {"BC_CFG_INVALID_FORMAT", RAD_E_FORMAT};
        case ConfigErrorCode::digest_mismatch: return {"BC_CFG_DIGEST_MISMATCH", RAD_E_FORMAT};
        case ConfigErrorCode::experiment_forbidden: return {"BC_CFG_EXPERIMENT_FORBIDDEN", RAD_E_SCHEMA};
        case ConfigErrorCode::unresolved_reference: return {"BC_CFG_UNRESOLVED_REFERENCE", RAD_E_NOTFOUND};
        case ConfigErrorCode::provenance_invalid: return {"BC_CFG_PROVENANCE_INVALID", RAD_E_SCHEMA};
        case ConfigErrorCode::calibration_invalid: return {"BC_CFG_CALIBRATION_INVALID", RAD_E_SCHEMA};
        case ConfigErrorCode::ownership_mismatch: return {"BC_CFG_OWNERSHIP_MISMATCH", RAD_E_SCHEMA};
        case ConfigErrorCode::plan_mismatch: return {"BC_CFG_PLAN_MISMATCH", RAD_E_SCHEMA};
        case ConfigErrorCode::admission_refused: return {"BC_CFG_ADMISSION_REFUSED", RAD_E_NOMEM};
        case ConfigErrorCode::io_failure: return {"BC_CFG_IO_FAILURE", RAD_E_IO};
        case ConfigErrorCode::abi_mismatch: return {"BC_CFG_ABI_MISMATCH", RAD_E_ABI};
        case ConfigErrorCode::invalid_state: return {"BC_CFG_INVALID_STATE", RAD_E_STATE};
    }
    return {"BC_CFG_INVALID_STATE", RAD_E_STATE};
}
}  // namespace

ConfigError make_error(ConfigErrorCode code, std::string path, std::string message, std::string hint) {
    ConfigError e;
    e.code = code;
    e.status = status_of(code);
    e.path = std::move(path);
    e.message = std::move(message);
    e.hint = std::move(hint);
    return e;
}

int status_of(ConfigErrorCode code) { return row(code).status; }
std::string_view machine_name(ConfigErrorCode code) { return row(code).name; }

std::string_view status_name(int status) {
    switch (status) {
        case RAD_E_NOMEM: return "RAD_E_NOMEM";
        case RAD_E_SCHEMA: return "RAD_E_SCHEMA";
        case RAD_E_ABI: return "RAD_E_ABI";
        case RAD_E_DUPLICATE: return "RAD_E_DUPLICATE";
        case RAD_E_INVAL: return "RAD_E_INVAL";
        case RAD_E_IO: return "RAD_E_IO";
        case RAD_E_FORMAT: return "RAD_E_FORMAT";
        case RAD_E_STATE: return "RAD_E_STATE";
        case RAD_E_NOTFOUND: return "RAD_E_NOTFOUND";
        case RAD_E_UNSUPPORTED: return "RAD_E_UNSUPPORTED";
        default: return "RAD_E_STATE";
    }
}

int exit_code_of(const ConfigError& e) {
    switch (e.code) {
        case ConfigErrorCode::io_failure: return 3;
        case ConfigErrorCode::admission_refused: return 4;
        case ConfigErrorCode::invalid_state: return 5;
        default: return 2;
    }
}

std::string render_human(std::string_view executable, const ConfigError& e) {
    std::string out = std::string(executable) + ": config: " + std::string(machine_name(e.code)) + " (" +
                      std::string(status_name(e.status)) + ", " + std::to_string(e.status) + ")\n";
    out += "  " + (e.path.empty() ? std::string() : e.path + ": ") + e.message + "\n";
    if (e.source_file) {
        out += "  source: " + *e.source_file;
        if (e.line) out += ":" + std::to_string(*e.line);
        if (e.line && e.column) out += ":" + std::to_string(*e.column);
        out += "\n";
    }
    if (!e.hint.empty()) out += "  hint: " + e.hint + "\n";
    return out;
}

std::string render_json(const ConfigError& e) {
    nlohmann::ordered_json j;
    j["code"] = machine_name(e.code);
    j["status"] = e.status;
    j["path"] = e.path;
    j["message"] = e.message;
    j["hint"] = e.hint;
    j["source_file"] = e.source_file ? nlohmann::ordered_json(*e.source_file) : nlohmann::ordered_json(nullptr);
    j["line"] = e.line ? nlohmann::ordered_json(*e.line) : nlohmann::ordered_json(nullptr);
    j["column"] = e.column ? nlohmann::ordered_json(*e.column) : nlohmann::ordered_json(nullptr);
    return j.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace);
}

}  // namespace bc::serving::v2
