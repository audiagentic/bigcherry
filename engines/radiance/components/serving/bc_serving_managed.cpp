/* bc_serving_managed.cpp -- the engine-independent half of the managed native entry (see bc_serving_managed.h): the flag
 * step, the process-global managed state and the chosen-cards check. Built into bc_serving_v2 so the manifest entry (rad0009)
 * links it in every engine build; the bind that needs the engine's Config is bc_serving_managed_bind.cpp (rad0070). */
#include "bc_serving_managed.h"

#include "document_bytes.h"
#include "engine_admission.h"
#include "strict_json.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {
namespace v2 = bc::serving::v2;

std::optional<bc::serving::EngineAdmission> g_admission;
std::atomic<bool> g_managed{false};
int g_admission_fd = -1;
std::atomic<bool> g_engine_exited{false};

std::string g_last_refusal;
}  // namespace

int bc_serving_managed_refuse(const std::string& why) {
    g_last_refusal = why;
    std::fprintf(stderr, "radiance: managed start: %s\n", why.c_str());
    return 2;
}
void bc_serving_managed_mark() { g_managed.store(true); }
int bc_serving_managed_admission_fd() { return g_admission_fd; }
const bc::serving::EngineAdmission* bc_serving_managed_admission() { return g_admission ? &*g_admission : nullptr; }

namespace {
int refuse(const std::string& why) { return bc_serving_managed_refuse(why); }
std::optional<std::string> value_of(int* argc, char*** argv, const char* flag, std::string* error) {
    std::optional<std::string> found;
    int w = 1;
    for (int i = 1; i < *argc; ++i) {
        if (std::strcmp((*argv)[i], flag) != 0) {
            (*argv)[w++] = (*argv)[i];
            continue;
        }
        if (found || i + 1 >= *argc || !*(*argv)[i + 1]) {
            *error = std::string(flag) + (found ? " given twice" : " needs a value");
            return std::nullopt;
        }
        found = (*argv)[++i];
    }
    (*argv)[w] = nullptr;
    *argc = w;
    return found;
}
}  // namespace

int bc_serving_managed_take_flag(int* argc, char*** argv) {
    bool present = false;
    for (int i = 1; i < *argc; ++i) present |= std::strcmp((*argv)[i], "--bc-admission-fd") == 0;
    if (!present) return 0;   // the stock entry: argv is not touched
    std::string error;
    const auto fd_text = value_of(argc, argv, "--bc-admission-fd", &error);
    if (!fd_text) return refuse(error);
    const auto nonce = value_of(argc, argv, "--bc-heat-launch-nonce", &error);
    if (!error.empty()) return refuse(error);
    const auto phase = value_of(argc, argv, "--bc-step-phase-dir", &error);
    if (!error.empty()) return refuse(error);

    int fd = -1;
    const char* end = fd_text->data() + fd_text->size();
    const auto [ptr, ec] = std::from_chars(fd_text->data(), end, fd);
    if (ec != std::errc{} || ptr != end || fd < 3) return refuse("--bc-admission-fd needs a descriptor greater than 2");
    try {
        const std::string text = v2::read_sealed_document(fd, "engine admission descriptor");
        const auto parsed = v2::parse_strict_json(text, "engine admission");
        if (!parsed) return refuse(v2::render_human("engine admission", parsed.error()));
        auto admission = bc::serving::engine_admission_from_json(parsed.value());
        if (!admission) return refuse(v2::render_human("engine admission", admission.error()));
        if (nonce) {
            const bool hex = nonce->size() == 32 && std::all_of(nonce->begin(), nonce->end(), [](char c) {
                return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
            });
            if (!hex) return refuse("--bc-heat-launch-nonce must be exactly 128 bits of lowercase hex");
            if (admission.value().measuring || !admission.value().histogram_file)
                return refuse("--bc-heat-launch-nonce requires a serving start with an authenticated histogram publisher");
        }
        if (phase) {
            namespace fs = std::filesystem;
            if (admission.value().measuring || !fs::path(*phase).is_absolute() || !fs::is_directory(*phase) ||
                fs::is_symlink(fs::symlink_status(*phase)))
                return refuse("--bc-step-phase-dir is valid only for a serving start and must name an existing safe absolute directory");
        }
        for (const auto& [name, value] : admission.value().env) ::setenv(name.c_str(), value.c_str(), 1);
        bc_serving_install_engine_state(admission.value(), nonce.value_or(""), phase.value_or(""));
        g_admission = std::move(admission.value());
        g_admission_fd = fd;
    } catch (const std::exception& failure) {
        return refuse(failure.what());
    }
    return 0;
}


const std::string& bc_serving_managed_last_refusal() { return g_last_refusal; }
bool bc_serving_managed() { return g_managed.load(); }
void bc_serving_managed_note_engine_exit() { g_engine_exited.store(true); }
bool bc_serving_managed_engine_exited() { return g_engine_exited.load(); }

std::vector<int> bc_serving_managed_usable(const std::vector<int>& single_architecture) {
    if (!g_admission) return single_architecture;
    std::vector<int> ranks(g_admission->bdfs.size());
    for (size_t i = 0; i < ranks.size(); ++i) ranks[i] = static_cast<int>(i);
    return ranks;
}

std::string bc_serving_managed_chosen(const std::vector<int>& chosen) {
    if (!g_admission) return {};
    std::vector<int> want(g_admission->bdfs.size());
    for (size_t i = 0; i < want.size(); ++i) want[i] = static_cast<int>(i);
    if (chosen == want) return {};
    const auto list = [](const std::vector<int>& v) {
        std::string out;
        for (int c : v) out += (out.empty() ? "" : ",") + std::to_string(c);
        return out;
    };
    return "the cards Fleet chose (" + list(chosen) + ") are not the admitted rank order (" + list(want) + ")";
}
