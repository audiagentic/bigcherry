#pragma once
/* BRVP06 item 5: a build runs against the ROCm SDK recorded in its metadata (rocm_root; its RUNPATH points at it), and bc-serve
 * refuses to start when the environment points the process at another SDK. The launcher's check is lib/sdk-check.sh; this is the
 * same rule for bc-serve, which is part of the same build (<build>/radiance_home/bin/bc-serve). */
#include <filesystem>
#include <optional>
#include <string>

namespace bc::serving {

/* The three environment values that can point a process at an SDK; empty = unset. */
struct SdkEnvironment {
    std::string rocm_path, hip_path, ld_library_path;
};
SdkEnvironment current_sdk_environment();

/* Why `exe` (the running bc-serve, resolved) must not run in `env`, or nothing when it may. The build is found from the
 * executable's place, <build>/radiance_home/bin/bc-serve, whose metadata <build>/bigcherry-build-metadata-radiance.json records
 * toolchain.rocm_root. An executable that is not inside a build (a component test build, a copy elsewhere) has no recorded SDK
 * and is not checked. Inside a build the metadata must record an installed rocm_root, ROCM_PATH and HIP_PATH must resolve to it
 * when set, and no LD_LIBRARY_PATH directory may carry ROCm runtime libraries from another install; nothing is overridden
 * silently. */
std::optional<std::string> sdk_refusal(const std::filesystem::path& exe, const SdkEnvironment& env);

}  // namespace bc::serving
