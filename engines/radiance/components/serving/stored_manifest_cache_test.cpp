// RSC12 S5: the static key and index are distinct from the strict-v2
// manifest digest, live-room admission and the per-start execution record.
// Every test is CPU-only, with real file bytes and deterministic failure.
#include "sha256.h"
#include "stored_manifest_cache.h"

#include <cassert>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef __clang__
#if __has_feature(address_sanitizer)
#define BC_TEST_ADDRESS_SANITIZER 1
#endif
#elif defined(__SANITIZE_ADDRESS__)
#define BC_TEST_ADDRESS_SANITIZER 1
#endif

namespace fs = std::filesystem;
namespace ss = bc::serving::session;
using Json = nlohmann::json;

namespace {
void expect(bool ok, const std::string& why) {
    if (!ok)
        throw std::runtime_error(why);
}
struct Temporary {
    fs::path root;
    Temporary() {
        std::string name = (fs::temp_directory_path() / "rsc12-stored-cache-XXXXXX").string();
        std::vector<char> chars(name.begin(), name.end());
        chars.push_back('\0');
        const char* made = ::mkdtemp(chars.data());
        if (!made)
            throw std::runtime_error("mkdtemp");
        root = made;
    }
    ~Temporary() {
        std::error_code unused;
        fs::remove_all(root, unused);
    }
};
void write(const fs::path& path, const std::string& data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out || !(out << data))
        throw std::runtime_error("write " + path.string());
}
ss::StaticResolutionInputs base() {
    ss::StaticResolutionInputs s;
    s.effective_values = {{"service", {{"max_context", 8192}}}, {"layout", {{"name", "fixture"}}}};
    s.ordered_devices = Json::array({{{"role_group", "engine_cards"},
                                      {"bdf", "0000:0d:00.0"},
                                      {"arch", "gfx1201"},
                                      {"hip_ordinal", 1},
                                      {"vram_total_mib", 24576}}});
    s.producer_digest = std::string(64, 'a');
    s.engine_digest = std::string(64, 'b');
    s.helper_digest = std::string(64, 'c');
    s.model_digest = std::string(64, 'd');
    s.power_profile = "stock";
    s.file_digests = {{"config.toml", std::string(64, 'e')},
                      {"heat.csv", std::string(64, 'f')},
                      {"calibration.toml", std::string(64, '0')},
                      {"experiment.json", std::string(64, '1')}};
    return s;
}
} // namespace

int main() {
    try {
        Temporary t;
        // FIPS 180-4 vectors and a multi-update block-boundary fixture guard
        // against changing the repository's existing manifest hash grammar.
        expect(bc::serving::sha256_hex("abc") ==
                   "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
               "incremental SHA-256 standard vector");
        expect(bc::serving::sha256_hex("") ==
                   "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
               "incremental SHA-256 empty vector");
        bc::serving::Sha256 stream;
        stream.update(std::string(63, 'a'));
        stream.update("b");
        stream.update(std::string(65, 'c'));
        expect(stream.hex_digest() ==
                   bc::serving::sha256_hex(std::string(63, 'a') + "b" + std::string(65, 'c')),
               "incremental SHA-256 must match flat hash across block boundaries");
        const fs::path model = t.root / "model.bin";
        const fs::path saved = t.root / "manifest-record.json";
        const fs::path index = ss::stored_index_path(t.root, "fixture");
        const fs::path work_root = t.root / "work";
        fs::create_directory(work_root);
        const fs::path another_run = work_root / "runs" / "new-start";
        fs::create_directories(another_run);
        const fs::path new_index = ss::stored_index_path(another_run, "different-config");
        // More than one block: changing weights past any header/table must
        // invalidate a complete artifact identity, without loading it all.
        std::string contents((2U * 1024U * 1024U) + 5U, 'm');
        write(model, contents);
        const std::string first = ss::complete_file_identity(model);
        expect(first == ss::complete_file_identity_bytes(contents),
               "identity from authenticated multi-megabyte bytes must match the streamed path identity");
        // Compare with the original chunk-accumulator wire digest, not a new
        // similarly named hash; persisted static keys must remain compatible.
        std::string old_accumulator = "bigcherry/rsc12/chunked-file-v1\n";
        constexpr size_t chunk_bytes = size_t{1024} * size_t{1024};
        for (size_t off = 0; off < contents.size(); off += chunk_bytes) {
            const size_t count = std::min(chunk_bytes, contents.size() - off);
            old_accumulator +=
                std::to_string(count) + ":" + bc::serving::sha256_hex(contents.substr(off, count)) + "\n";
        }
        expect(first ==
                   bc::serving::sha256_hex("bytes=" + std::to_string(contents.size()) + "\n" + old_accumulator),
               "streamed full fingerprint must preserve historic domain-separated chunk digest");
        contents.back() = 'x';
        write(model, contents);
        const std::string second = ss::complete_file_identity(model);
        expect(first != second, "late model payload byte must change full artifact identity");

        auto s = base();
        const std::string original = ss::static_resolution_key(s);
        expect(original == ss::static_resolution_key(base()), "equal static inputs mean equal key");
        const auto changed = [&](const std::string& what) {
            expect(ss::static_resolution_key(s) != original, what + " must invalidate stored key");
            s = base();
        };
        s.effective_values["service"]["max_context"] = 16384;
        changed("effective settings/variant");
        s.ordered_devices[0]["bdf"] = "0000:0e:00.0";
        changed("ordered BDF");
        s.ordered_devices[0]["arch"] = "gfx1100";
        changed("GPU architecture");
        s.ordered_devices[0]["vram_total_mib"] = 32768;
        changed("VRAM total");
        s.producer_digest[0] = 'f';
        changed("bc-serve binary");
        s.engine_digest[0] = 'f';
        changed("engine/kernel/runtime binary");
        s.helper_digest[0] = 'f';
        changed("helper binary");
        s.model_digest = second;
        changed("model artifact, including tensor bytes");
        s.power_profile = "energy";
        changed("power profile");
        s.file_digests["config.toml"][0] = 'a';
        changed("config file");
        s.file_digests["heat.csv"][0] = 'a';
        changed("heat");
        s.file_digests["calibration.toml"][0] = 'a';
        changed("calibration/refit evidence");
        s.file_digests["experiment.json"][0] = 'a';
        changed("lab experiment bytes");
        s.ordered_devices[0]["vram_available_mib"] = 42;
        bool unexpected_live = false;
        try {
            (void)ss::static_resolution_key(s);
        } catch (const std::exception&) {
            unexpected_live = true;
        }
        expect(unexpected_live, "live VRAM must not enter a static fingerprint");
        s = base();

        write(saved, "unchanged strict-v2 manifest bytes\n");
        const std::string digest = ss::complete_file_identity_bytes("unchanged strict-v2 manifest bytes\n");
        expect(digest == ss::complete_file_identity(saved),
               "fingerprint over authenticated bytes must preserve the persisted chunked identity");
        // Fresh run/config must reuse the shared fingerprint. An index
        // only binds a saved manifest to its original model digest.
        unsigned full_model_reads = 0;
        const auto counting_model_reader = [&](const fs::path& source) {
            ++full_model_reads;
            return ss::complete_file_identity(source);
        };
        const auto cache_file = ss::model_identity_cache_path(model, work_root);
        const ss::ModelIdentity model_first = ss::cached_model_identity(model, work_root, counting_model_reader);
        expect(full_model_reads == 1 && model_first.digest == second && fs::is_regular_file(cache_file),
               "first start fingerprints all bytes and writes the shared cache file");
        ss::write_stored_index(index, {original, saved.filename().string(), digest, model_first.digest});
        expect(ss::stored_manifest_for(t.root, "fixture", original) == saved,
               "named index must resolve using the existing manifest fingerprint");
        ss::write_stored_index(new_index, {original, saved.filename().string(), digest, model_first.digest});
        const auto model_hit = ss::cached_model_identity(model, work_root, counting_model_reader);
        expect(full_model_reads == 1 && model_hit.digest == model_first.digest,
               "fresh run dir and config must make zero model reader calls");
        const auto same_attempt = ss::cached_model_identity(model, work_root, counting_model_reader, &model_first);
        expect(full_model_reads == 1 && same_attempt.digest == model_first.digest,
               "verified unchanged model remains a zero-read identity");
        // Same-size rewrite with restored mtime must still invalidate ctime.
        const auto prior_mtime = fs::last_write_time(model);
        write(model, contents);
        fs::last_write_time(model, prior_mtime);
        const auto model_changed = ss::cached_model_identity(model, work_root, counting_model_reader, &model_first);
        expect(full_model_reads == 2 && model_changed.digest == second,
               "same-size ctime change must recompute despite restored mtime");
        write(model, contents + "changed");
        const auto model_changed_size = ss::cached_model_identity(model, work_root, counting_model_reader);
        expect(full_model_reads == 3 && model_changed_size.digest != second,
               "changed size must recompute and replace the shared cache");
        write(cache_file, "{bad json");
        bool cache_refused = false;
        try {
            (void)ss::cached_model_identity(model, work_root, counting_model_reader);
        } catch (const std::exception&) {
            cache_refused = true;
        }
        expect(cache_refused && full_model_reads == 3,
               "malformed shared cache must be refused, not trusted or recomputed");
        fs::remove(cache_file);
        fs::create_symlink(model, cache_file);
        cache_refused = false;
        try {
            (void)ss::cached_model_identity(model, work_root, counting_model_reader);
        } catch (const std::exception&) {
            cache_refused = true;
        }
        expect(cache_refused && full_model_reads == 3, "symlink cache must be rejected");
        fs::remove(cache_file);

        // Forked readers race through a cold cache key; independent atomic
        // writers may all win, but every published record must be complete.
        std::array<pid_t, 6> writers{};
        for (auto& pid : writers) {
            pid = ::fork();
            expect(pid >= 0, "fork shared cache writer");
            if (pid == 0) {
                try {
                    (void)ss::cached_model_identity(model, work_root, [&](const fs::path& source) {
                        ::usleep(50000);
                        return ss::complete_file_identity(source);
                    });
                    ::_exit(0);
                } catch (...) {
                    ::_exit(40);
                }
            }
        }
        for (const pid_t pid : writers) {
            int status = 0;
            expect(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                   "parallel cache writers must each succeed");
        }
        const auto after_race = ss::cached_model_identity(model, work_root, counting_model_reader);
        expect(full_model_reads == 3 && after_race.digest == model_changed_size.digest,
               "concurrent writes must yield parseable JSON with a correct cache hit");
        // Retargeting a model symlink mid-read must refuse, not cache old bytes.
        const fs::path alias = t.root / "model-alias.bin";
        const fs::path alternate = t.root / "alternate.bin";
        write(alternate, "different model bytes");
        fs::remove(cache_file);
        fs::create_symlink(model, alias);
        bool swapped_refused = false;
        try {
            (void)ss::cached_model_identity(alias, work_root, [&](const fs::path& source) {
                fs::remove(alias);
                fs::create_symlink(alternate, alias);
                return ss::complete_file_identity(source);
            });
        } catch (const std::exception&) {
            swapped_refused = true;
        }
        expect(swapped_refused && !fs::exists(cache_file),
               "retargeted model alias must refuse without caching a stale digest");
        ss::write_stored_index(index, {original, saved.filename().string(), digest});
        const auto found = ss::stored_manifest_for(t.root, "fixture", original);
        expect(found && *found == saved, "equal static key must reuse unchanged manifest");
        expect(!ss::stored_manifest_for(t.root, "fixture", std::string(64, 'a')),
               "different static key must re-resolve and write a new manifest");
        expect(ss::read_stored_index(index).digest == digest, "index must preserve exact original-byte digest");

        write(saved, "tampered manifest\n");
        bool refused = false;
        try {
            (void)ss::stored_manifest_for(t.root, "fixture", original);
        } catch (const std::exception&) {
            refused = true;
        }
        expect(refused, "equal key but changed original bytes must fail closed, not rewrite");
        write(saved, "unchanged strict-v2 manifest bytes\n");
        ss::write_stored_index(index, {std::string(64, 'a'), saved.filename().string(), digest});
        expect(!ss::stored_manifest_for(t.root, "fixture", original),
               "new different index must replace cache selection without touching old manifest");
        expect(fs::file_size(saved) == std::string("unchanged strict-v2 manifest bytes\n").size(),
               "index replacement must not modify stored bytes");
        write(index, "{\"version\":1,\"manifest\":\"../escape.json\",\"digest\":\"x\",\"key\":\"x\"}");
        refused = false;
        try {
            (void)ss::read_stored_index(index);
        } catch (const std::exception&) {
            refused = true;
        }
        expect(refused, "unsafe cache index must refuse even under root");
        // RED P2-7: a corrupt sparse index must be rejected before allocating
        // its full apparent size. Limit the child virtual address space so the
        // test remains safe even when the original unbounded read regresses.
        fs::resize_file(index, 512ULL * 1024ULL * 1024ULL);
        const pid_t child = ::fork();
        expect(child >= 0, "fork bounded index verifier");
        if (child == 0) {
            // ASan needs a large virtual address space for shadow mappings,
            // so preserve the 64 MiB cap only in the unsanitized regression.
#ifndef BC_TEST_ADDRESS_SANITIZER
            const rlimit budget{64ULL * 1024ULL * 1024ULL, 64ULL * 1024ULL * 1024ULL};
            if (::setrlimit(RLIMIT_AS, &budget) != 0)
                ::_exit(30);
#endif
            try {
                (void)ss::read_stored_index(index);
                ::_exit(31);
            } catch (const std::exception& e) {
                ::_exit(std::string(e.what()).find("exceeds 8 KiB") != std::string::npos ? 0 : 32);
            }
        }
        int exit_status = 0;
        expect(::waitpid(child, &exit_status, 0) == child, "wait on bounded index verifier");
        expect(WIFEXITED(exit_status) && WEXITSTATUS(exit_status) == 0,
               "oversized sparse index must refuse at 8193 bytes, without allocating it");
        std::cout << "RSC12 static manifest reuse/rewrite/refusals passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RSC12 static manifest cache test: " << error.what() << '\n';
        return 1;
    }
}
