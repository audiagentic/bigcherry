// RSC12 S5: the static key and index are distinct from the strict-v2
// manifest digest, live-room admission and the per-start execution record.
// Every test is CPU-only, with real file bytes and deterministic failure.
#include "stored_manifest_cache.h"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

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
        const fs::path model = t.root / "model.bin";
        const fs::path saved = t.root / "manifest-record.json";
        const fs::path index = ss::stored_index_path(t.root, "fixture");
        // More than one block: changing weights past any header/table must
        // invalidate a complete artifact identity, without loading it all.
        std::string contents((2U * 1024U * 1024U) + 5U, 'm');
        write(model, contents);
        const std::string first = ss::complete_file_identity(model);
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
        const std::string digest = ss::complete_file_identity(saved);
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
        std::cout << "RSC12 static manifest reuse/rewrite/refusals passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RSC12 static manifest cache test: " << error.what() << '\n';
        return 1;
    }
}
