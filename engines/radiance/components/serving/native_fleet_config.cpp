#define _GNU_SOURCE 1
#include "native_fleet_config.h"

#include "fleet/fleet.h"
#include "v2/document_bytes.h"

#include <ryml.hpp>

#include <charconv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace bc::serving {

namespace {

struct XBigcherryError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
[[noreturn]] void on_basic(ryml::csubstr msg, ryml::ErrorDataBasic const&, void*) {
    throw XBigcherryError(std::string(msg.str, msg.len));
}
[[noreturn]] void on_parse(ryml::csubstr msg, ryml::ErrorDataParse const&, void*) {
    throw XBigcherryError(std::string(msg.str, msg.len));
}
[[noreturn]] void on_visit(ryml::csubstr msg, ryml::ErrorDataVisit const&, void*) {
    throw XBigcherryError(std::string(msg.str, msg.len));
}

// YAML 1.2 core typing of one plain scalar: null, bool, integer, float; a quoted scalar is always a string.
nlohmann::json typed_scalar(ryml::ConstNodeRef node) {
    const std::string text(node.val().str, node.val().len);
    if (node.is_val_quoted()) return text;
    if (text.empty() || text == "null" || text == "Null" || text == "NULL" || text == "~") return nullptr;
    if (text == "true" || text == "True" || text == "TRUE") return true;
    if (text == "false" || text == "False" || text == "FALSE") return false;
    const char* begin = text.data();
    const char* end = begin + text.size();
    std::int64_t integer = 0;
    if (const auto [ptr, error] = std::from_chars(begin, end, integer); error == std::errc{} && ptr == end)
        return integer;
    // A decimal float: digits with one point or an exponent, no leading plus-less words such as "nan" or "inf".
    if (text.find_first_of("0123456789") != std::string::npos &&
        text.find_first_not_of("+-0123456789.eE") == std::string::npos) {
        char* parsed_end = nullptr;
        const double value = std::strtod(begin, &parsed_end);
        if (parsed_end == end && std::isfinite(value)) return value;
    }
    return text;
}

nlohmann::json yaml_json(ryml::ConstNodeRef node, const std::string& path) {
    if (node.is_map()) {
        nlohmann::json object = nlohmann::json::object();
        for (ryml::ConstNodeRef child : node.children()) {
            const std::string key(child.key().str, child.key().len);
            if (object.contains(key)) throw XBigcherryError(path + "." + key + " is given twice");
            object[key] = yaml_json(child, path + "." + key);
        }
        return object;
    }
    if (node.is_seq()) {
        nlohmann::json array = nlohmann::json::array();
        std::size_t index = 0;
        for (ryml::ConstNodeRef child : node.children())
            array.push_back(yaml_json(child, path + "[" + std::to_string(index++) + "]"));
        return array;
    }
    return typed_scalar(node);
}

}  // namespace

// The top-level x-bigcherry subtree of the sealed YAML as JSON, null when the file has none. Radiance's parser skips
// x-* keys and keeps its tree private, so the same bytes are parsed once more here with the same library (with
// anchors and merge keys resolved as it does): the one YAML parser; bc-serve receives JSON and decodes it against the
// closed schema. A duplicate key anywhere below x-bigcherry is refused, naming its YAML path.
nlohmann::json x_bigcherry_subtree(const std::string& yaml_utf8) {
    try {
        ryml::Callbacks callbacks;
        callbacks.set_error_basic(&on_basic).set_error_parse(&on_parse).set_error_visit(&on_visit);
        ryml::EventHandlerTree handler(callbacks);
        ryml::Parser parser(&handler);
        ryml::Tree tree(callbacks);
        ryml::parse_in_arena(&parser, ryml::to_csubstr("native config"), ryml::to_csubstr(yaml_utf8), &tree);
        tree.resolve();
        if (tree.empty() || !tree.crootref().has_children()) return nullptr;
        ryml::ConstNodeRef root = tree.crootref();
        if (root.is_stream()) root = root.first_child();
        if (!root.is_map()) return nullptr;
        for (ryml::ConstNodeRef child : root.children())
            if (std::string_view(child.key().str, child.key().len) == "x-bigcherry")
                return yaml_json(child, "x-bigcherry");
    } catch (const XBigcherryError& error) {
        throw std::invalid_argument(std::string("native config x-bigcherry: ") + error.what());
    }
    return nullptr;
}

nlohmann::json yaml_document_json(const std::string& yaml_utf8, const char* name) {
    try {
        ryml::Callbacks callbacks;
        callbacks.set_error_basic(&on_basic).set_error_parse(&on_parse).set_error_visit(&on_visit);
        ryml::EventHandlerTree handler(callbacks);
        ryml::Parser parser(&handler);
        ryml::Tree tree(callbacks);
        ryml::parse_in_arena(&parser, ryml::to_csubstr(name), ryml::to_csubstr(yaml_utf8), &tree);
        tree.resolve();
        if (tree.empty() || !tree.crootref().has_children()) throw std::invalid_argument(std::string(name) + ": empty document");
        ryml::ConstNodeRef root = tree.crootref();
        if (root.is_stream()) root = root.first_child();
        if (!root.is_map()) throw std::invalid_argument(std::string(name) + ": the document must be a mapping");
        return yaml_json(root, name);
    } catch (const XBigcherryError& error) {
        throw std::invalid_argument(std::string(name) + ": " + error.what());
    }
}

NativeFleetConfig NativeFleetConfig::read(const std::filesystem::path& authored,
                                          const std::filesystem::path& resolution_root) {
    return NativeFleetConfig(NativeConfigSnapshot::read(authored, resolution_root), authored.string());
}

NativeFleetConfig NativeFleetConfig::replay(NativeConfigSnapshot snapshot) {
    return NativeFleetConfig(std::move(snapshot), "spec.native_config.yaml_utf8");
}

NativeFleetConfig NativeFleetConfig::replay(const v2::NativeConfigView& admitted) {
    return replay(NativeConfigSnapshot::admitted(std::string(admitted.yaml_utf8),
                                                 std::string(admitted.identity.sha256),
                                                 std::string(admitted.identity.resolution_root)));
}

NativeFleetConfig::NativeFleetConfig(NativeConfigSnapshot snapshot, const std::string& origin)
    : snapshot_(std::move(snapshot)) {
    snapshot_.require_execution_root();
    if (rad::fleet::parse(snapshot_.yaml_utf8(), origin, &fleet_) != RAD_OK)
        throw std::invalid_argument("Radiance refused the native fleet YAML: " + origin);
    if (fleet_.models.size() != 1)
        throw std::invalid_argument("managed native fleet YAML requires exactly one model");
    if (const std::string unsupported = rad::fleet::unsupported(fleet_); !unsupported.empty())
        throw std::invalid_argument("Radiance build does not support managed fleet settings: " + unsupported);
    // Managed manifests construct one Engine from the model Config. These
    // settings have no effect without the upstream Fleet lifecycle.
    const auto& model = fleet_.models.front();
    const rad::fleet::FleetConfig defaults;
    const std::pair<bool, const char*> fleet_only[] = {
        {fleet_.host_mib != defaults.host_mib, "memory.host-mib"},
        {fleet_.disk_tier != defaults.disk_tier, "memory.disk-tier"},
        {fleet_.load_timeout_ms != defaults.load_timeout_ms, "scheduler.load-timeout"},
        {fleet_.drain_timeout_ms != defaults.drain_timeout_ms, "scheduler.drain-timeout"},
        {fleet_.keepalive_ms != defaults.keepalive_ms, "scheduler.keepalive"},
        {fleet_.stuck_exit_ms != defaults.stuck_exit_ms, "scheduler.stuck-exit"},
        {fleet_.warm_up != defaults.warm_up, "scheduler.warm-up"},
        {!model.aliases.empty(), "aliases"},
        {model.priority != 0, "priority"},
        {model.pin, "pin"},
        {model.ttl_ms != 0, "ttl"},
        {model.flex != rad::fleet::FlexSpec{}, "flex"},
        {model.park != rad::fleet::ParkSpec{}, "park"},
    };
    for (const auto& [configured, option] : fleet_only)
        if (configured)
            throw std::invalid_argument(std::string("managed single-engine YAML cannot apply fleet-only setting: ") +
                                        option);
    // Native provenance includes inherited and aliased server options. Checking
    // the option, not its resolved value, also permits environment references.
    if (model.cfg.given.contains("--api-key"))
        throw std::invalid_argument("managed native fleet YAML requires server.api-key-env for credentials");
}

std::optional<int> inspect_native_fleet_config_command(int argc, char** argv) {
    constexpr const char* command = "--bc-inspect-native-config";
    if (argc < 2 || argv[1] == nullptr || std::strcmp(argv[1], command) != 0) return std::nullopt;
    const bool with_profiles = argc >= 5 && argv[3] != nullptr && std::strcmp(argv[3], "--bc-profile-fd") == 0;
    const bool with_site = with_profiles && argc == 7 && argv[5] != nullptr && std::strcmp(argv[5], "--bc-site-fd") == 0;
    if ((argc != 3 && !(with_profiles && (argc == 5 || with_site))) || argv[2] == nullptr ||
        (with_profiles && argv[4] == nullptr) || (with_site && argv[6] == nullptr)) {
        std::cerr << "radiance: " << command
                  << " requires one inherited sealed descriptor holding the raw YAML [--bc-profile-fd FD for the power profile file"
                     " [--bc-site-fd FD for this host's power site file]]\n";
        return 2;
    }

#if defined(__linux__)
    try {
        int fd = -1;
        const char* end = argv[2] + std::strlen(argv[2]);
        const auto [ptr, error] = std::from_chars(argv[2], end, fd);
        if (error != std::errc{} || ptr != end || fd < 3)
            throw std::invalid_argument("descriptor must be an integer greater than 2");

        nlohmann::json power_profiles = nullptr;
        if (with_profiles) {
            int profile_fd = -1;
            const char* profile_end = argv[4] + std::strlen(argv[4]);
            const auto [profile_ptr, profile_error] = std::from_chars(argv[4], profile_end, profile_fd);
            if (profile_error != std::errc{} || profile_ptr != profile_end || profile_fd < 3)
                throw std::invalid_argument("profile descriptor must be an integer greater than 2");
            power_profiles = yaml_document_json(v2::read_sealed_document(profile_fd, "power profile descriptor"),
                                                "power-profiles");
        }
        nlohmann::json power_site = nullptr;
        if (with_site) {
            int site_fd = -1;
            const char* site_end = argv[6] + std::strlen(argv[6]);
            const auto [site_ptr, site_error] = std::from_chars(argv[6], site_end, site_fd);
            if (site_error != std::errc{} || site_ptr != site_end || site_fd < 3)
                throw std::invalid_argument("site descriptor must be an integer greater than 2");
            power_site = yaml_document_json(v2::read_sealed_document(site_fd, "power site descriptor"), "power-site");
        }
        NativeFleetConfig managed = NativeFleetConfig::replay(NativeConfigSnapshot::from_raw_fd(fd));
        const auto& fleet = managed.fleet();
        const auto& cfg = fleet.models.front().cfg;
        auto stated = [&](const char* key, const std::string& value) -> nlohmann::json {
            return cfg.given.contains(key) ? nlohmann::json(value) : nlohmann::json(nullptr);
        };

        const nlohmann::json report = {
            {"protocol", "bigcherry.native-fleet-inspect/v2"},
            {"snapshot", {{"sha256", managed.snapshot().sha256()},
                           {"resolution_root", managed.snapshot().resolution_root()}}},
            {"engine", {{"abi_version", RAD_ABI_VERSION},
                         {"model", cfg.model},
                         {"radiance_home", cfg.radiance_home},
                         {"kernels", cfg.kernel_hierarchy},
                         {"listener", {{"host", cfg.host}, {"port", cfg.port}}},
                         {"tp", cfg.tp},
                         {"tp_wire", stated("--tp-wire", cfg.tp_wire_exact ? "exact" : "wht6")},
                         {"cards", fleet.cards},
                         {"rank_devices", cfg.rank_devices},
                         {"max_context", cfg.max_ctx},
                         {"max_sequences", cfg.max_seqs},
                         {"x_bigcherry", x_bigcherry_subtree(managed.snapshot().yaml_utf8())},
                         {"power_profiles", power_profiles},
                         {"power_site", power_site},
                         // The tuning options the authored YAML states, as values the strict reader parsed (null: absent).
                         {"tuning", {{"p2p", stated("--p2p", cfg.p2p == rad::RAD_P2P_ON    ? "on"
                                                             : cfg.p2p == rad::RAD_P2P_OFF ? "off" : "auto")},
                                      {"pipelined_prefill", stated("--pipelined-prefill",
                                                                   cfg.pipelined_prefill == 1   ? "on"
                                                                   : cfg.pipelined_prefill == 2 ? "off" : "auto")},
                                      {"pipelined_prefill_min_rows",
                                       cfg.given.contains("--pipelined-prefill-min-rows")
                                           ? nlohmann::json(cfg.pipelined_prefill_min_rows) : nlohmann::json(nullptr)},
                                      {"prefill_governor", stated("--prefill-governor", cfg.prefill_governor)},
                                      {"tp_ownership", stated("--tp-ownership", cfg.tp_ownership)},
                                      {"tune_pin", stated("--tune-pin", cfg.tune_pin)},
                                      {"attn_prefill_fp8", stated("--attn-prefill-fp8",
                                          cfg.attn_prefill_fp8 == RAD_ATTN_FP8_QK ? "qk" :
                                          cfg.attn_prefill_fp8 == RAD_ATTN_FP8_PV ? "pv" :
                                          cfg.attn_prefill_fp8 == (RAD_ATTN_FP8_QK | RAD_ATTN_FP8_PV) ? "both" : "off")},
                                      {"busy_prefill_chunk", stated("--busy-prefill-chunk",
                                          cfg.busy_prefill_chunk < 0 ? "auto" : cfg.busy_prefill_chunk == 0 ? "off" :
                                          std::to_string(cfg.busy_prefill_chunk))},
                                      {"state_cache_dtype", stated("--state-cache-dtype", cfg.state_cache_dtype)},
                                      {"tp_wire_min_kb", cfg.given.contains("--tp-wire-min-kb")
                                          ? nlohmann::json(cfg.tp_wire_min_kb) : nlohmann::json(nullptr)}}},
                         {"settings", {{"max_batched_tokens", cfg.max_tok},
                                       {"checkpoint_interval", cfg.checkpoint_interval},
                                       {"placement", cfg.placement},
                                       {"ngram_placement", rad::table_placement_name(cfg.ngram_placement)},
                                       {"kv_dtype", cfg.kv_cache_dtype},
                                       {"speculative_tokens", cfg.n_spec},
                                       {"rope_scaling", cfg.rope_scaling},
                                       {"prefix_resume", cfg.prefix_exact ? "exact" : "nearest"},
                                       {"prefix_cache", cfg.prefix_cache},
                                       {"prefill_interleave", cfg.prefill_interleave_set && cfg.prefill_interleave >= 0
                                           ? nlohmann::json(cfg.prefill_interleave) : nlohmann::json(nullptr)}}}}}
        };
        std::cout << report.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "radiance: " << command << ": " << error.what() << '\n';
        return 2;
    }
#else
    std::cerr << "radiance: " << command << " is supported only on Linux\n";
    return 2;
#endif
}

}  // namespace bc::serving
