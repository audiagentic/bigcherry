#include "sdk_check.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

namespace bc::serving {
namespace fs = std::filesystem;

SdkEnvironment current_sdk_environment() {
    auto get = [](const char* name) {
        const char* v = std::getenv(name);
        return v ? std::string(v) : std::string();
    };
    return {get("ROCM_PATH"), get("HIP_PATH"), get("LD_LIBRARY_PATH")};
}

std::optional<std::string> sdk_refusal(const fs::path& exe, const SdkEnvironment& env) {
    std::error_code ec;
    const fs::path real = fs::weakly_canonical(exe, ec);
    // <build>/radiance_home/bin/bc-serve
    if (ec || real.parent_path().filename() != "bin" || real.parent_path().parent_path().filename() != "radiance_home")
        return std::nullopt;
    const fs::path build = real.parent_path().parent_path().parent_path();
    const fs::path meta = build / "bigcherry-build-metadata-radiance.json";
    if (!fs::is_regular_file(meta, ec)) return std::nullopt;

    std::string root;
    try {
        std::ifstream in(meta, std::ios::binary);
        std::ostringstream text;
        text << in.rdbuf();
        const nlohmann::json doc = nlohmann::json::parse(text.str());
        for (const auto& pair : doc.at("toolchain"))
            if (pair.is_array() && pair.size() == 2 && pair[0] == "rocm_root" && pair[1].is_string()) root = pair[1].get<std::string>();
    } catch (const std::exception& e) {
        return "sdk-check: " + meta.string() + " cannot be read (" + e.what() + ")";
    }
    if (root.empty()) return "sdk-check: " + meta.string() + " records no rocm_root; rebuild with a named ROCm stack";
    if (!fs::is_directory(root, ec)) return "sdk-check: the ROCm SDK this build records (" + root + ") is not installed";
    const fs::path sdk = fs::canonical(root, ec);

    for (const auto& [name, value] : {std::pair<const char*, const std::string&>{"ROCM_PATH", env.rocm_path},
                                      std::pair<const char*, const std::string&>{"HIP_PATH", env.hip_path}}) {
        if (value.empty()) continue;
        const fs::path given = fs::weakly_canonical(value, ec);
        if (given != sdk)
            return std::string("sdk-check: ") + name + "=" + value + " is not the SDK this build records (" + sdk.string() + "); unset it";
    }
    std::istringstream dirs(env.ld_library_path);
    for (std::string dir; std::getline(dirs, dir, ':');) {
        if (dir.empty()) continue;
        if (!fs::exists(fs::path(dir) / "libamdhip64.so", ec) && !fs::exists(fs::path(dir) / "libhsa-runtime64.so", ec)) continue;
        const fs::path where = fs::weakly_canonical(dir, ec);
        const std::string w = where.string(), s = sdk.string() + "/";
        if (w.compare(0, s.size(), s) != 0)
            return "sdk-check: LD_LIBRARY_PATH entry " + dir + " carries ROCm libraries outside the recorded SDK (" + sdk.string() + ")";
    }
    return std::nullopt;
}

}  // namespace bc::serving
