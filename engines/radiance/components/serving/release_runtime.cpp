#include "release_runtime.h"

#include "release_install.h"

#include <stdexcept>
#include <string>

namespace bc::serving {
namespace fs = std::filesystem;

std::optional<fs::path> installed_prefix(const fs::path& exe) {
    std::error_code ec;
    const fs::path real = fs::canonical(exe, ec);
    if (ec || real.filename() != "bc-serve" || real.parent_path().filename() != "bin" ||
        real.parent_path().parent_path().filename() != "radiance_home")
        return std::nullopt;
    fs::path prefix = real.parent_path().parent_path().parent_path();
    if (prefix.parent_path().filename() != "releases")
        return std::nullopt;
    return prefix;
}

fs::path verify_installed_prefix(const fs::path& exe, const fs::path& sdk) {
    const auto prefix = installed_prefix(exe);
    if (!prefix)
        throw std::runtime_error("installed release refusal: executable is not in a release prefix");
    const fs::path root = prefix->parent_path().parent_path();
    try {
        const auto verified = bc::release::verify_current(root, sdk);
        if (verified.at("current").get<std::string>() != prefix->filename().string())
            throw std::runtime_error("installed release refusal: executable is not the verified current tag");
    } catch (const bc::release::Error& error) {
        throw std::runtime_error("installed release refusal BC_RELEASE_" +
                                 std::to_string(static_cast<int>(error.code())) + ": " + error.what());
    }
    return *prefix;
}

} // namespace bc::serving
