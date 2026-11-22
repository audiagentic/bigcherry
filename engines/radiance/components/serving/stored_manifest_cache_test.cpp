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
    s.effective_values = {{"service", {{"max_context", 8192}}}, {"layout", {{"name", "fixture"}}},
                        {"engine", {{"model", "/models/fixture.rad"}, {"radiance_home", "/installed/build-a/radiance_home"}}}};
    s.ordered_devices = Json::array({{{"role_group", "engine_cards"},
                                      {"bdf", "0000:0d:00.0"},
                                      {"arch", "gfx1201"},
                                      {"hip_ordinal", 1},
                                      {"vram_total_mib", 24576}}});
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
        const fs::path saved = t.root / "run-1" / "manifest-record.json";
        fs::create_directories(saved.parent_path());
        const fs::path work_root = t.root / "work";
        fs::create_directory(work_root);
        const fs::path index = ss::named_store_index(work_root, "fixture");
        expect(index.parent_path().parent_path() == work_root / "cache" &&
                   index != ss::named_store_index(work_root, "other-config"),
               "the named store lives under the work root, one index per config");
        std::string contents((2U * 1024U * 1024U) + 5U, 'm');
        write(model, contents);
        const std::string first = ss::complete_file_identity(model);
        expect(first == ss::complete_file_identity_bytes(contents),
               "identity from authenticated multi-megabyte bytes must match the streamed path identity");
        std::string old_accumulator = "bigcherry/rsc12/chunked-file-v1\n";
        constexpr size_t chunk_bytes = size_t{1024} * size_t{1024};
        for (size_t off = 0; off < contents.size(); off += chunk_bytes) {
            const size_t count = std::min(chunk_bytes, contents.size() - off);
            old_accumulator +=
                std::to_string(count) + ":" + bc::serving::sha256_hex(contents.substr(off, count)) + "\n";
        }
        expect(first ==
                   bc::serving::sha256_hex("bytes=" + std::to_string(contents.size()) + "\n" + old_accumulator),
               "streamed full fingerprint must preserve the chunk digest used for small files");

        // A model is identified by path, size and mtime: no byte of it is read.
        const std::string model_first = ss::model_stat_identity(model);
        expect(model_first == ss::model_stat_identity(model), "an untouched model keeps its identity");
        const auto prior_mtime = fs::last_write_time(model);
        contents.back() = 'x';
        write(model, contents);
        fs::last_write_time(model, prior_mtime);
        expect(ss::model_stat_identity(model) == model_first,
               "same path, size and mtime: the identity is stat-based and does not read the bytes");
        fs::last_write_time(model, prior_mtime + std::chrono::seconds(5));
        const std::string model_touched = ss::model_stat_identity(model);
        expect(model_touched != model_first, "a changed mtime changes the model identity");
        write(model, contents + "more");
        expect(ss::model_stat_identity(model) != model_touched, "a changed size changes the model identity");
        const fs::path huge = t.root / "huge.bin";
        {
            std::ofstream sparse(huge, std::ios::binary);
            sparse << 'x';
        }
        fs::resize_file(huge, 64ULL * 1024ULL * 1024ULL * 1024ULL);
        expect(ss::model_stat_identity(huge).size() == 64U, "a 64 GiB model is identified without reading it");
        const fs::path alias = t.root / "model-alias.bin";
        fs::create_symlink(model, alias);
        expect(ss::model_stat_identity(alias) == ss::model_stat_identity(model),
               "an alias of the model resolves to the same identity");
        bool missing_refused = false;
        try {
            (void)ss::model_stat_identity(t.root / "absent.bin");
        } catch (const std::exception&) {
            missing_refused = true;
        }
        expect(missing_refused, "a missing model has no identity");
        const std::string model_id = ss::model_stat_identity(model);

        auto s = base();
        s.model_digest = model_id;
        const auto base_with_model = [&] {
            auto b = base();
            b.model_digest = model_id;
            return b;
        };
        const std::string original = ss::static_resolution_key(s);
        expect(original == ss::static_resolution_key(base_with_model()), "equal static inputs mean equal key");
        const auto changed = [&](const std::string& what) {
            expect(ss::static_resolution_key(s) != original, what + " must invalidate stored key");
            s = base_with_model();
        };
        const auto unchanged = [&](const std::string& what) {
            expect(ss::static_resolution_key(s) == original, what + " must NOT invalidate the stored key");
            s = base_with_model();
        };
        s.effective_values["service"]["max_context"] = 16384;
        changed("effective settings/variant");
        s.effective_values["engine"]["model"] = "/models/other.rad";
        changed("model path in the config");
        s.ordered_devices[0]["bdf"] = "0000:0e:00.0";
        changed("ordered BDF");
        s.ordered_devices[0]["arch"] = "gfx1100";
        changed("GPU architecture");
        s.ordered_devices[0]["vram_total_mib"] = 32768;
        changed("VRAM total");
        s.model_digest = model_touched;
        changed("model size or mtime");
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
        // What a different installed build changes, and the run's own transport, is provenance, not identity.
        s.effective_values["engine"]["radiance_home"] = "/installed/other-build/radiance_home";
        unchanged("the installed runtime path");
        s = [&] {
            auto projected = base_with_model();
            projected.effective_values["handoff"] = {{"shm", "/r3-serve-first"}};
            projected.effective_values["memory"] = {{"host_pool_mib", 9}, {"host_kv_mib", 8}};
            return projected;
        }();
        const std::string with_transport = ss::static_resolution_key(s);
        s.effective_values["handoff"] = {{"shm", "/r3-serve-second"}};
        s.effective_values["memory"] = {{"host_pool_mib", 1}, {"host_kv_mib", 2}};
        expect(ss::static_resolution_key(s) == with_transport,
               "the per-start transport name and derived host pool/KV never enter the key");
        s = base_with_model();
        s.ordered_devices[0]["vram_available_mib"] = 42;
        bool unexpected_live = false;
        try {
            (void)ss::static_resolution_key(s);
        } catch (const std::exception&) {
            unexpected_live = true;
        }
        expect(unexpected_live, "live VRAM must not enter a static fingerprint");
        s = base_with_model();

        write(saved, "unchanged strict-v2 manifest bytes\n");
        const std::string digest = ss::complete_file_identity_bytes("unchanged strict-v2 manifest bytes\n");
        expect(digest == ss::complete_file_identity(saved),
               "fingerprint over authenticated bytes must preserve the persisted chunked identity");

        // The store is reached from any later run directory: it names the run that wrote the manifest.
        ss::StoredManifestIndex entry{original, saved.filename().string(), digest, model_id};
        entry.directory = saved.parent_path().string();
        entry.input_digests = ss::static_input_digests(s);
        ss::write_stored_index(index, entry);
        using Outcome = ss::PlanReuseOutcome;
        const auto now_inputs = ss::static_input_digests(s);
        auto outcome = [&]() {
            return ss::lookup_stored_manifest(index, original, now_inputs);
        };
        expect(ss::read_stored_index(index).input_digests == now_inputs, "per-input digests round-trip");
        {
            const auto hit = outcome();
            expect(hit.outcome == Outcome::hit && hit.manifest == saved,
                   "a later start in another run dir finds the manifest the earlier start stored");
            const std::string line = ss::plan_reuse_line(hit);
            expect(line.find("stored plan reused") != std::string::npos && line.find('\n') == std::string::npos &&
                       line.find("no stored plan") == std::string::npos,
                   "a hit prints one reuse line");
        }
        {
            // A different key: the line names the inputs that moved, from this start's digests.
            auto other = s;
            other.power_profile = "other-profile";
            const auto changed = ss::lookup_stored_manifest(index, ss::static_resolution_key(other),
                                                            ss::static_input_digests(other));
            expect(changed.outcome == Outcome::static_key_changed && changed.inputs_recorded &&
                       changed.changed_inputs == std::vector<std::string>{"power_profile"},
                   "a changed key names exactly the input that moved");
            const std::string line = ss::plan_reuse_line(changed);
            expect(line == "bc-serve: no stored plan reused (static key changed: power_profile); measuring",
                   "a miss prints one line naming the reason: " + line);
            auto other_file = s;
            other_file.file_digests["/some/dir/site.toml"] = std::string(64, 'e');
            const auto added = ss::lookup_stored_manifest(index, ss::static_resolution_key(other_file),
                                                          ss::static_input_digests(other_file));
            expect(added.changed_inputs.size() == 1 && added.changed_inputs[0].rfind("file:site.toml@", 0) == 0 &&
                       ss::plan_reuse_line(added).find("/some/dir") == std::string::npos,
                   "a new file input is named by base name, never by its path");
            const auto unrecorded = ss::lookup_stored_manifest(index, std::string(64, 'a'));
            expect(unrecorded.outcome == Outcome::static_key_changed && !unrecorded.inputs_recorded &&
                       ss::plan_reuse_line(unrecorded).find("predates") != std::string::npos,
                   "a key change without current digests says so and does not invent leaves");
        }
        expect(ss::read_stored_index(index).digest == digest && ss::read_stored_index(index).directory == entry.directory,
               "index must preserve exact original-byte digest and run directory");
        {
            const auto absent = ss::lookup_stored_manifest(
                work_root / "cache" / "named-configs" / "never-written.json", original, now_inputs);
            expect(absent.outcome == Outcome::absent &&
                       ss::plan_reuse_line(absent) == "bc-serve: no stored plan reused (no stored entry); measuring",
                   "no index is absent");
        }
        {
            ss::PlanReuse refused;
            refused.outcome = Outcome::live_admission_refused;
            refused.detail = "not enough room";
            expect(ss::plan_reuse_line(refused) ==
                       "bc-serve: no stored plan reused (live admission refused: not enough room); measuring",
                   "live admission refusal line");
            refused.outcome = Outcome::carried_plan_unusable;
            refused.detail = "plan does not fit";
            expect(ss::plan_reuse_line(refused).find("stored plan unusable: plan does not fit") != std::string::npos,
                   "carried plan line");
            expect(ss::plan_reuse_json(refused).at("outcome") == "carried_plan_unusable", "outcome is recorded by name");
        }

        // A damaged store is a miss, never a refusal to start: the next measured manifest replaces it.
        write(saved, "tampered manifest\n");
        expect(outcome().outcome == Outcome::integrity_mismatch, "changed original bytes are an integrity miss, not a refusal");
        write(saved, "unchanged strict-v2 manifest bytes\n");
        fs::remove(saved);
        expect(outcome().outcome == Outcome::manifest_missing, "a removed run dir is a missing-manifest miss");
        write(saved, "unchanged strict-v2 manifest bytes\n");
        expect(outcome().outcome == Outcome::hit && outcome().manifest == saved, "restoring the bytes restores the hit");
        write(index, "{bad json");
        expect(outcome().outcome == Outcome::index_unreadable, "a corrupt index is an unreadable-entry miss");
        ss::write_stored_index(index, entry);
        ss::StoredManifestIndex relative = entry;
        relative.directory = "relative/run";
        bool relative_refused = false;
        try {
            ss::write_stored_index(index, relative);
        } catch (const std::exception&) {
            relative_refused = true;
        }
        expect(relative_refused, "the stored run directory must be absolute");
        ss::write_stored_index(index, {std::string(64, 'a'), saved.filename().string(), digest});
        expect(outcome().outcome == Outcome::static_key_changed,
               "a newer index replaces the selection without touching the old manifest");
        expect(fs::file_size(saved) == std::string("unchanged strict-v2 manifest bytes\n").size(),
               "index replacement must not modify stored bytes");
        write(index, "{\"version\":1,\"manifest\":\"../escape.json\",\"digest\":\"x\",\"key\":\"x\"}");
        bool refused = false;
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
