// The engine identity digests the runtime once per process and still notices every change. CPU-only, real files.
#include "plan_cache.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace fs = std::filesystem;
namespace sv = bc::serving;

namespace {
void expect(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
void write(const fs::path& p, const std::string& bytes) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary | std::ios::trunc) << bytes;
}
}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / ("bc-engine-identity-" + std::to_string(::getpid()));
    try {
        fs::remove_all(root);
        const fs::path binary = root / "bin" / "engine", home = root / "home";
        write(binary, "engine-bytes");
        write(home / "kernels" / "k.bin", "kernel-a");
        write(home / "helpers" / "h.bin", "helper-a");

        const uint64_t base = sv::engine_identity_hash_runs();
        const std::string first = sv::engine_identity(binary, home);
        expect(sv::engine_identity_hash_runs() == base + 1, "the first call hashes the runtime");
        for (int i = 0; i < 5; ++i)
            expect(sv::engine_identity(binary, home) == first, "an unchanged runtime keeps its identity");
        expect(sv::engine_identity_hash_runs() == base + 1, "an unchanged runtime is hashed once, not per check");

        // Same size, content changed: the change time moves, so the runtime is hashed again and the identity differs.
        write(home / "kernels" / "k.bin", "kernel-b");
        const std::string kernel_changed = sv::engine_identity(binary, home);
        expect(kernel_changed != first, "a changed kernel changes the identity");
        expect(sv::engine_identity_hash_runs() == base + 2, "a changed file is hashed again");

        write(binary, "engine-bytes-2");
        const std::string engine_changed = sv::engine_identity(binary, home);
        expect(engine_changed != kernel_changed, "a changed engine binary changes the identity");

        write(home / "helpers" / "extra.bin", "new");
        const std::string added = sv::engine_identity(binary, home);
        expect(added != engine_changed, "an added runtime file changes the identity");
        fs::remove(home / "helpers" / "extra.bin");
        expect(sv::engine_identity(binary, home) == engine_changed, "removing it restores the earlier identity");

        write(home / "kernels" / "k.bin", "kernel-a");
        write(binary, "engine-bytes");
        expect(sv::engine_identity(binary, home) == first, "restored bytes give the original identity, not a stale one");

        bool refused = false;
        try {
            sv::engine_identity(root / "missing", home);
        } catch (const std::runtime_error&) {
            refused = true;
        }
        expect(refused, "a missing engine binary still refuses");
        fs::remove_all(root);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "engine identity test failed: " << error.what() << '\n';
        fs::remove_all(root);
        return 1;
    }
}
