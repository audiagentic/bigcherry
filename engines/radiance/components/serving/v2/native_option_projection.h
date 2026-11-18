// RUM04 replacement seam: today the single serving-v2 field registry projects
// strictly admitted BigCherry intent to Radiance's native flags. RUM04 replaces
// this projection with upstream fleet parsing of the immutable original runtime
// bundle. This file owns NO option/env names; generated kFields is the table.
//
// Header-only C++17 so the patched engine and card-free tests share exactly
// the same translation implementation without a second serializer.
#pragma once

#include <nlohmann/json.hpp>

#include <set>
#include <string>
#include <utility>
#include <vector>

namespace bc::serving::v2 {

struct NativeOptionProjection {
    std::vector<std::pair<std::string, std::string>> flags;
    std::set<std::string> switches;
    std::string error;

    // The actual native argv shape: switches have no following argument.
    std::vector<std::string> argv() const {
        std::vector<std::string> result;
        for (const auto& flag : flags) {
            result.push_back(flag.first);
            if (!switches.count(flag.first))
                result.push_back(flag.second);
        }
        return result;
    }
};

inline bool native_option_text(const nlohmann::json& value, std::string* text) {
    if (value.is_string()) {
        *text = value.get<std::string>();
        return true;
    }
    if (value.is_number_integer()) {
        *text = value.dump();
        return true;
    }
    if (value.is_array()) {
        std::string joined;
        for (const auto& element : value) {
            if (!element.is_string())
                return false;
            if (!joined.empty())
                joined += ',';
            joined += element.get<std::string>();
        }
        *text = std::move(joined);
        return true;
    }
    return false;
}

template <typename Fields>
NativeOptionProjection project_native_options(const nlohmann::json& manifest, const Fields& fields,
                                              const std::set<std::string>& known_options) {
    NativeOptionProjection out;
    for (const auto& field : fields) {
        if (field.radiance_option.empty() || field.path.rfind("/spec/", 0) != 0)
            continue;
        const nlohmann::json::json_pointer ptr{std::string(field.path)};
        if (!manifest.contains(ptr))
            continue;

        const std::string option(field.radiance_option);
        if (!known_options.count(option)) {
            out.error = "\"" + std::string(field.path) + "\" is bound to " + option +
                        ", which is not a radiance option";
            out.flags.clear();
            out.switches.clear();
            return out;
        }
        const auto& value = manifest.at(ptr);
        if (field.radiance_switch) {
            out.flags.emplace_back(option, "");
            out.switches.insert(option);
            continue;
        }
        std::string argument;
        if (!native_option_text(value, &argument)) {
            out.error = "\"" + std::string(field.path) +
                        "\" has a value that is not a string, an integer or an array of strings";
            out.flags.clear();
            out.switches.clear();
            return out;
        }
        out.flags.emplace_back(option, std::move(argument));
    }
    return out;
}

} // namespace bc::serving::v2
