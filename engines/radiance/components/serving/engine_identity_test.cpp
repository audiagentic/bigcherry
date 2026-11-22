// The engine identity digests the runtime once per process, and once per work root while no file moves, and still
// notices every change. CPU-only, real files. A "fresh process" is a real one: this binary run again as a child.
#include "plan_cache.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <sys/stat.h>
#include <unistd.h>
#include <utime.h>

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

struct ChildResult {
    std::string identity;
    uint64_t hashed = 0;
};

// engine_identity in a new process, with the identity store at `work` (empty: none); what it returned and how many
// times it hashed the runtime.
ChildResult fresh_process(const char* self, const fs::path& binary, const fs::path& home, const fs::path& work) {
    const std::string cmd = std::string("'") + self + "' --child '" + binary.string() + "' '" + home.string() + "' '" +
                            work.string() + "'";
    FILE* pipe = ::popen(cmd.c_str(), "r");
    expect(pipe != nullptr, "the child starts");
    char buf[256]{};
    const bool read = std::fgets(buf, sizeof(buf), pipe) != nullptr;
    expect(::pclose(pipe) == 0 && read, "the child succeeds");
    std::istringstream line(buf);
    ChildResult r;
    line >> r.identity >> r.hashed;
    return r;
}

int child(char** argv) {
    const fs::path work = argv[4];
    if (!work.empty()) sv::set_engine_identity_store(work);
    const std::string id = sv::engine_identity(argv[2], argv[3]);
    std::cout << id << ' ' << sv::engine_identity_hash_runs() << '\n';
    return 0;
}

// Let the file system's clock tick, so a change after this call moves a file's stamps (a same-size rewrite in the same
// tick as the earlier write would otherwise carry the same times).
void tick() { ::usleep(20000); }

fs::path persisted_file(const fs::path& work) {
    fs::path found;
    for (const auto& e : fs::recursive_directory_iterator(work))
        if (e.is_regular_file()) {
            expect(found.empty(), "one small file per runtime home");
            found = e.path();
        }
    expect(!found.empty(), "the digest is persisted under the work root");
    return found;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "--child") return child(argv);
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

        // The digest outlives the process: a fresh process under the same work root reads it and hashes nothing.
        const fs::path work = root / "work";
        fs::create_directories(work);
        const char* self = argv[0];
        tick();
        const auto stored = fresh_process(self, binary, home, work);
        expect(stored.identity == first, "the persisted path gives the digest the hash gives");
        expect(stored.hashed == 1, "the first fresh process hashes and stores");
        const auto reread = fresh_process(self, binary, home, work);
        expect(reread.identity == first && reread.hashed == 0, "a fresh process reads the persisted digest, hashing zero files");
        (void)persisted_file(work);

        // No work root: only the per-process memo, so every fresh process hashes.
        expect(fresh_process(self, binary, home, "").hashed == 1, "no work root: no persistence");
        expect(fresh_process(self, binary, home, "").hashed == 1, "no work root: still no persistence");

        // Content changed (same size), an added file, a removed file and an mtime-only touch each hash once, then the
        // new state is stored and read again.
        auto changes = [&](const char* what, auto&& mutate) {
            mutate();
            tick();
            const auto changed = fresh_process(self, binary, home, work);
            expect(changed.hashed == 1, what);
            const auto again = fresh_process(self, binary, home, work);
            expect(again.hashed == 0 && again.identity == changed.identity, "the changed state is stored");
            return changed.identity;
        };
        const std::string content = changes("changed content hashes", [&] { write(home / "kernels" / "k.bin", "kernel-b"); });
        expect(content != first, "changed content changes the digest");
        const std::string with_extra = changes("an added file hashes", [&] { write(home / "helpers" / "extra.bin", "new"); });
        expect(with_extra != content, "an added file changes the digest");
        const std::string without_extra = changes("a removed file hashes", [&] { fs::remove(home / "helpers" / "extra.bin"); });
        expect(without_extra == content, "removing it restores the earlier digest");
        const std::string touched = changes("an mtime-only touch hashes", [&] {
            const utimbuf t{1000000000, 1000000000};
            expect(::utime((home / "helpers" / "h.bin").c_str(), &t) == 0, "touch");
        });
        expect(touched == content, "a touch alone does not change the digest, only re-checks it");

        // A corrupt or foreign persisted file is ignored, hashed over, and replaced.
        const fs::path stored_file = persisted_file(work);
        write(stored_file, "{ not json");
        expect(fresh_process(self, binary, home, work).hashed == 1, "a corrupt persisted file is ignored");
        expect(fresh_process(self, binary, home, work).hashed == 0, "and replaced by a good one");
        std::string foreign;
        { std::ifstream in(stored_file); std::ostringstream s; s << in.rdbuf(); foreign = s.str(); }
        const auto at = foreign.find(home.string());
        expect(at != std::string::npos, "the persisted file names its runtime home");
        foreign.replace(at, home.string().size(), (root / "other").string());
        write(stored_file, foreign);
        expect(fresh_process(self, binary, home, work).hashed == 1, "a persisted file for another runtime home is ignored");

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
