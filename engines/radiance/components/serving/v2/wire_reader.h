#pragma once
/* A strict reader for one closed JSON object, shared by the typed process documents (HelperSettings, EngineAdmission):
 * every member is asked for by name (a missing one is a fault) and done() refuses the rest. */
#include "error.h"

#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

namespace bc::serving::wire {
using Json = nlohmann::json;
struct Fault {
    v2::ConfigErrorCode code;
    std::string path, message;
};

// Reads one closed JSON object: every member is asked for by name (a missing one is a fault) and done() refuses the rest.
class Wire {
  public:
    Wire(const Json& object, std::string path) : o_(object), path_(std::move(path)) {
        if (!o_.is_object()) throw Fault{v2::ConfigErrorCode::invalid_type, path_, "expected an object"};
    }
    const Json& at(const char* key) {
        if (!o_.contains(key)) throw Fault{v2::ConfigErrorCode::missing_field, path_ + "." + key, "missing"};
        used_.insert(key);
        return o_.at(key);
    }
    bool boolean(const char* key) {
        const Json& v = at(key);
        if (!v.is_boolean()) bad(key, "a boolean");
        return v.get<bool>();
    }
    std::string text(const char* key) {
        const Json& v = at(key);
        if (!v.is_string()) bad(key, "a string");
        return v.get<std::string>();
    }
    uint64_t u64(const char* key, uint64_t max = UINT64_MAX) {
        const Json& v = at(key);
        if (!v.is_number_unsigned() || v.get<uint64_t>() > max) bad(key, "an unsigned integer in range");
        return v.get<uint64_t>();
    }
    uint32_t u32(const char* key) { return static_cast<uint32_t>(u64(key, UINT32_MAX)); }
    double real(const char* key) {
        const Json& v = at(key);
        if (!v.is_number()) bad(key, "a number");
        return v.get<double>();
    }
    template <class T>
    std::vector<T> list(const char* key) {
        const Json& v = at(key);
        if (!v.is_array()) bad(key, "an array");
        std::vector<T> out;
        for (const Json& e : v) {
            if constexpr (std::is_same_v<T, double>) {
                if (!e.is_number()) bad(key, "an array of numbers");
                out.push_back(e.get<double>());
            } else {
                if (!e.is_number_unsigned() || e.get<uint64_t>() > std::numeric_limits<T>::max())
                    bad(key, "an array of unsigned integers in range");
                out.push_back(static_cast<T>(e.get<uint64_t>()));
            }
        }
        return out;
    }
    std::string path(const char* key) const { return path_ + "." + key; }
    void done() const {
        for (auto it = o_.begin(); it != o_.end(); ++it)
            if (!used_.count(it.key())) throw Fault{v2::ConfigErrorCode::unknown_field, path_ + "." + it.key(), "unknown member"};
    }

  private:
    [[noreturn]] void bad(const char* key, const char* want) const {
        throw Fault{v2::ConfigErrorCode::invalid_type, path_ + "." + key, std::string("expected ") + want};
    }
    const Json& o_;
    std::string path_;
    std::set<std::string> used_;
};

}  // namespace bc::serving::wire
