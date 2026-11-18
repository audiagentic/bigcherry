#include "sdk_check.h"

#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace bc::serving {
namespace fs = std::filesystem;

SdkEnvironment current_sdk_environment() {
    auto get = [](const char* name) {
        const char* v = std::getenv(name);
        return v ? std::string(v) : std::string();
    };
    return {get("ROCM_PATH"), get("HIP_PATH"), get("LD_LIBRARY_PATH")};
}

std::optional<fs::path> recorded_sdk_root(const fs::path& exe) {
    std::error_code ec;
    const fs::path real = fs::weakly_canonical(exe, ec);
    if (ec || real.parent_path().filename() != "bin" ||
        real.parent_path().parent_path().filename() != "radiance_home")
        return std::nullopt;
    const fs::path build = real.parent_path().parent_path().parent_path();
    const fs::path meta = build / "bigcherry-build-metadata-radiance.json";
    if (!fs::is_regular_file(meta, ec))
        throw std::runtime_error("sdk-check: build/release metadata missing: " + meta.string());
    std::string root;
    try {
        std::ifstream in(meta, std::ios::binary);
        if (!in)
            throw std::runtime_error("cannot open");
        std::ostringstream text;
        text << in.rdbuf();
        if (!in.good() && !in.eof())
            throw std::runtime_error("cannot read");
        const nlohmann::json doc = nlohmann::json::parse(text.str());
        for (const auto& pair : doc.at("toolchain"))
            if (pair.is_array() && pair.size() == 2 && pair[0] == "rocm_root" && pair[1].is_string())
                root = pair[1].get<std::string>();
    } catch (const std::exception& e) {
        throw std::runtime_error("sdk-check: " + meta.string() + " cannot be read (" + e.what() + ")");
    }
    if (root.empty())
        throw std::runtime_error("sdk-check: " + meta.string() +
                                 " records no rocm_root; rebuild with a named ROCm stack");
    if (!fs::is_directory(root, ec))
        throw std::runtime_error("sdk-check: the ROCm SDK this build records (" + root + ") is not installed");
    fs::path sdk = fs::canonical(root, ec);
    if (ec)
        throw std::runtime_error("sdk-check: recorded ROCm SDK cannot be resolved: " + root);
    return sdk;
}

std::optional<std::string> sdk_refusal(const fs::path& exe, const SdkEnvironment& env) {
    std::optional<fs::path> selected;
    try {
        selected = recorded_sdk_root(exe);
    } catch (const std::exception& e) {
        return e.what();
    }
    if (!selected)
        return std::nullopt;
    const fs::path& sdk = *selected;
    std::error_code ec;

    for (const auto& [name, value] : {std::pair<const char*, const std::string&>{"ROCM_PATH", env.rocm_path},
                                      std::pair<const char*, const std::string&>{"HIP_PATH", env.hip_path}}) {
        if (value.empty())
            continue;
        const fs::path given = fs::weakly_canonical(value, ec);
        if (given != sdk)
            return std::string("sdk-check: ") + name + "=" + value + " is not the SDK this build records (" +
                   sdk.string() + "); unset it";
    }
    std::istringstream dirs(env.ld_library_path);
    for (std::string dir; std::getline(dirs, dir, ':');) {
        if (dir.empty())
            continue;
        if (!fs::exists(fs::path(dir) / "libamdhip64.so", ec) &&
            !fs::exists(fs::path(dir) / "libhsa-runtime64.so", ec))
            continue;
        const fs::path where = fs::weakly_canonical(dir, ec);
        const std::string w = where.string(), s = sdk.string() + "/";
        if (w.compare(0, s.size(), s) != 0)
            return "sdk-check: LD_LIBRARY_PATH entry " + dir +
                   " carries ROCm libraries outside the recorded SDK (" + sdk.string() + ")";
    }
    return std::nullopt;
}

} // namespace bc::serving
