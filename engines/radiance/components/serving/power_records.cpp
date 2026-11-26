#include "power_io.h"
#include "power_transaction.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <limits>
#include <memory>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace bc::serving::power {
namespace {
using Json = nlohmann::json;
using Code = v2::ConfigErrorCode;
constexpr size_t kMaximumJournalBytes = size_t{1024} * 1024U;
constexpr int64_t kIjsonMax = 9007199254740991LL;

v2::ConfigError error(Code code, std::string path, std::string message) {
    return v2::make_error(code, std::move(path), std::move(message));
}

using OwnedFd = detail::FileDescriptor;

v2::Result<OwnedFd> directory(const fs::path& path, bool create) {
    if (create) {
        std::error_code ec;
        fs::create_directories(path, ec);
        if (ec)
            return error(Code::io_failure, path.string(), "create directory: " + ec.message());
    }
    OwnedFd fd(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    if (!fd)
        return error(Code::io_failure, path.string(), "directory unavailable or symlinked");
    return fd;
}

v2::Status durable_write(const fs::path& path, const Json& document) {
    auto dir = directory(path.parent_path(), true);
    if (!dir)
        return dir.error();
    const auto serialized = document.dump(2) + "\n";
    const auto name = path.filename().string();
    const auto clock = std::chrono::steady_clock::now().time_since_epoch().count();
    std::string temporary;
    OwnedFd out;
    for (unsigned attempt = 0; attempt < 100U; ++attempt) {
        temporary = "." + name + "." + std::to_string(::getpid()) + "." + std::to_string(clock) + "." +
                    std::to_string(attempt);
        out = OwnedFd(::openat(dir.value().get(), temporary.c_str(),
                               O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
        if (out)
            break;
        if (errno != EEXIST)
            break;
    }
    if (!out)
        return error(Code::io_failure, path.string(), "cannot create temporary durable record");
    const auto cleanup = [&] { (void)::unlinkat(dir.value().get(), temporary.c_str(), 0); };
    size_t offset = 0;
    while (offset < serialized.size()) {
        const auto written = ::write(out.get(), serialized.data() + offset, serialized.size() - offset);
        if (written < 0 && errno == EINTR)
            continue;
        if (written <= 0) {
            cleanup();
            return error(Code::io_failure, path.string(), "durable record write failed");
        }
        offset += static_cast<size_t>(written);
    }
    if (::fsync(out.get()) != 0) {
        cleanup();
        return error(Code::io_failure, path.string(), "durable record fsync failed");
    }
    out = OwnedFd{};
    if (::renameat(dir.value().get(), temporary.c_str(), dir.value().get(), name.c_str()) != 0) {
        cleanup();
        return error(Code::io_failure, path.string(), "durable record rename failed");
    }
    if (::fsync(dir.value().get()) != 0)
        return error(Code::io_failure, path.string(), "durable directory fsync failed");
    return std::nullopt;
}

v2::Result<Json> durable_read(const fs::path& path) {
    auto dir = directory(path.parent_path(), false);
    if (!dir)
        return dir.error();
    OwnedFd fd(::openat(dir.value().get(), path.filename().c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
    if (!fd)
        return error(Code::io_failure, path.string(), "record unavailable or symlinked");
    struct stat st{};
    if (::fstat(fd.get(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        static_cast<uint64_t>(st.st_size) > kMaximumJournalBytes)
        return error(Code::invalid_format, path.string(), "record must be a bounded regular file");
    std::string bytes;
    bytes.reserve(static_cast<size_t>(st.st_size));
    char buffer[4096];
    for (;;) {
        const auto n = ::read(fd.get(), buffer, sizeof(buffer));
        if (n < 0 && errno == EINTR)
            continue;
        if (n < 0)
            return error(Code::io_failure, path.string(), "cannot read journal");
        if (n == 0)
            break;
        bytes.append(buffer, static_cast<size_t>(n));
        if (bytes.size() > kMaximumJournalBytes)
            return error(Code::invalid_format, path.string(), "power journal is oversized");
    }
    try {
        return Json::parse(bytes);
    } catch (const std::exception& e) {
        return error(Code::invalid_format, path.string(), std::string("invalid JSON: ") + e.what());
    }
}

std::string event_label(EventKind kind) {
    switch (kind) {
    case EventKind::prepared:
        return "prepared";
    case EventKind::touched:
        return "touched";
    case EventKind::applied:
        return "applied";
    case EventKind::apply_failed:
        return "apply_failed";
    case EventKind::restored:
        return "restored";
    case EventKind::restore_failed:
        return "restore_failed";
    case EventKind::restore_refused:
        return "restore_refused";
    case EventKind::rollback:
        return "rollback";
    }
    return "invalid";
}
v2::Result<EventKind> parse_event(std::string_view label) {
    for (auto value :
         {EventKind::prepared, EventKind::touched, EventKind::applied, EventKind::apply_failed,
          EventKind::restored, EventKind::restore_failed, EventKind::restore_refused, EventKind::rollback})
        if (label == event_label(value))
            return value;
    return error(Code::invalid_format, "power.events", "unknown power event");
}
std::string moment_label(Moment kind) {
    switch (kind) {
    case Moment::start:
        return "start";
    case Moment::post_apply:
        return "post_apply";
    case Moment::pre_restore:
        return "pre_restore";
    case Moment::stop:
        return "stop";
    case Moment::gfxoff:
        return "gfxoff";
    }
    return "invalid";
}

std::string_view observation_label(ObservationState state) {
    switch (state) {
    case ObservationState::observed:
        return "observed";
    case ObservationState::applied:
        return "applied";
    case ObservationState::restored:
        return "restored";
    case ObservationState::unavailable:
        return "unavailable";
    case ObservationState::unresolved:
        return "unresolved";
    case ObservationState::gfxoff_active:
        return "active";
    case ObservationState::gfxoff_absent:
        return "absent";
    case ObservationState::gfxoff_ambiguous:
        return "ambiguous";
    case ObservationState::gfxoff_not_applicable:
        return "not_applicable";
    case ObservationState::gfxoff_needs_context:
        return "needs_idle_engine_context";
    }
    return "invalid";
}
Json observed_json(const ObservedCard& card) {
    const auto field = [&](ControlId control) {
        const auto& value = card.values.at(static_cast<size_t>(control));
        return value ? Json(*value) : Json(nullptr);
    };
    Json result = {{"bdf", card.pci.value},
                   {"vddgfx_offset_mv", field(ControlId::voltage)},
                   {"sclk_max_mhz", field(ControlId::shader)},
                   {"mclk_max_mhz", field(ControlId::memory)},
                   {"power_cap_uw", field(ControlId::board_cap)}};
    if (card.values.at(static_cast<size_t>(ControlId::shader_offset)))
        result["sclk_offset_mhz"] = field(ControlId::shader_offset);
    if (!card.identity.empty())
        result["identity"] = card.identity;
    return result;
}
Json tuning_json(const Tuning& tuning, bool include_nulls) {
    Json result = Json::object();
    const auto put = [&](const char* label, std::optional<int64_t> value) {
        if (value)
            result[label] = *value;
        else if (include_nulls)
            result[label] = nullptr;
    };
    put("vddgfx_offset_mv",
        tuning.voltage_offset ? std::optional<int64_t>(tuning.voltage_offset->value) : std::nullopt);
    put("sclk_max_mhz",
        tuning.shader_ceiling ? std::optional<int64_t>(tuning.shader_ceiling->value) : std::nullopt);
    if (tuning.shader_offset)
        put("sclk_offset_mhz", tuning.shader_offset->value);
    put("mclk_max_mhz",
        tuning.memory_ceiling ? std::optional<int64_t>(tuning.memory_ceiling->value) : std::nullopt);
    put("power_cap_w", tuning.board_cap ? std::optional<int64_t>(tuning.board_cap->value) : std::nullopt);
    return result;
}
v2::Result<Tuning> parse_tuning(const Json& value) {
    if (!value.is_object())
        return error(Code::invalid_format, "power.setting", "power settings are not an object");
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (it.key() != "vddgfx_offset_mv" && it.key() != "sclk_max_mhz" && it.key() != "mclk_max_mhz" &&
            it.key() != "power_cap_w" && it.key() != "sclk_offset_mhz")
            return error(Code::invalid_format, "power.setting", "unknown journal control");
    }
    const auto field = [&](const char* name) -> v2::Result<std::optional<int64_t>> {
        if (!value.contains(name) || value.at(name).is_null())
            return std::optional<int64_t>{};
        if (!value.at(name).is_number_integer())
            return error(Code::invalid_format, "power.setting", "noninteger power value");
        auto number = value.at(name).get<int64_t>();
        if (number > kIjsonMax || number < -kIjsonMax)
            return error(Code::invalid_format, "power.setting", "value outside safe integer range");
        return std::optional<int64_t>{number};
    };
    auto voltage = field("vddgfx_offset_mv");
    auto shader = field("sclk_max_mhz");
    auto offset = field("sclk_offset_mhz");
    auto memory = field("mclk_max_mhz");
    auto power = field("power_cap_w");
    if (!voltage)
        return voltage.error();
    if (!shader)
        return shader.error();
    if (!offset)
        return offset.error();
    if (offset.value() && shader.value())
        return error(Code::invalid_format, "power.setting", "mutually exclusive shader controls");
    if (!memory)
        return memory.error();
    if (!power)
        return power.error();
    if (shader.value().value_or(1) <= 0 || memory.value().value_or(1) <= 0 || power.value().value_or(1) <= 0 ||
        power.value().value_or(1) > kIjsonMax / 1000000)
        return error(Code::invalid_format, "power.setting", "invalid positive power control");
    Tuning result;
    if (voltage.value().has_value())
        result.voltage_offset = Millivolts{voltage.value().value_or(0)};
    if (shader.value().has_value())
        result.shader_ceiling = Megahertz{shader.value().value_or(0)};
    if (offset.value().has_value())
        result.shader_offset = Megahertz{*offset.value()};
    if (memory.value().has_value())
        result.memory_ceiling = Megahertz{memory.value().value_or(0)};
    if (power.value().has_value())
        result.board_cap = Watts{power.value().value_or(0)};
    return result;
}
Json original_json(const CardSnapshot& state) {
    Json result = Json::object();
    const auto field = [&](const char* key, ControlId id) {
        const auto& observed = state.at(id).current;
        result[key] = observed ? Json(*observed) : Json(nullptr);
    };
    field("vddgfx_offset_mv", ControlId::voltage);
    field("sclk_max_mhz", ControlId::shader);
    field("mclk_max_mhz", ControlId::memory);
    field("power_cap_uw", ControlId::board_cap);
    if (state.at(ControlId::shader_offset).current)
        field("sclk_offset_mhz", ControlId::shader_offset);
    result["power_cap_default_uw"] = nullptr;
    return result;
}
v2::Status load_original(CardSnapshot& snapshot, const Json& record) {
    if (!record.is_object())
        return error(Code::invalid_format, "power.before", "original is not an object");
    for (auto [name, kind] : {std::pair{"vddgfx_offset_mv", ControlId::voltage},
                              {"sclk_max_mhz", ControlId::shader},
                              {"mclk_max_mhz", ControlId::memory},
                              {"power_cap_uw", ControlId::board_cap},
                              {"sclk_offset_mhz", ControlId::shader_offset}}) {
        if (kind == ControlId::shader_offset && !record.contains(name))
            continue; // old journal: live signed control will refuse restoration
        if (!record.contains(name))
            return error(Code::invalid_format, "power.before", "missing original power field");
        if (record.at(name).is_null())
            continue;
        if (!record.at(name).is_number_integer())
            return error(Code::invalid_format, "power.before", "invalid original power field");
        const auto number = record.at(name).get<int64_t>();
        if (number > kIjsonMax || number < -kIjsonMax)
            return error(Code::invalid_format, "power.before", "original outside safe integer range");
        snapshot.controls[static_cast<size_t>(kind)].current = number;
    }
    return std::nullopt;
}
Json events_json(const std::vector<Event>& history) {
    Json array = Json::array();
    for (const auto& event : history) {
        Json item = {{"kind", event_label(event.kind)}};
        if (event.pci)
            item["bdf"] = event.pci->value;
        if (!event.reason.empty())
            item["reason"] = event.reason;
        array.push_back(std::move(item));
    }
    return array;
}
Json journal_json(const Journal& journal) {
    Json cards = Json::array();
    for (const auto& saved : journal.devices) {
        cards.push_back(
            {{"bdf", saved.policy.device.pci.value},
             {"role", saved.policy.device.role.label()},
             {"hwmon", saved.before.hwmon ? Json("/sys/bus/pci/devices/" + saved.policy.device.pci.value +
                                                 "/hwmon/" + *saved.before.hwmon)
                                          : Json(nullptr)},
             {"setting", tuning_json(saved.policy.tuning, true)},
             {"before", original_json(saved.before)}});
        if (saved.before.at(ControlId::shader_offset).current)
            cards.back()["identity"] = saved.before.identity;
    }
    Json touched = Json::array();
    for (const auto& pci : journal.touched)
        touched.push_back(pci.value);
    return {{"schema_version", 1},
            {"run_id", journal.run_id},
            {"manifest_digest", journal.manifest_digest.value},
            {"profile_digest", journal.profile_digest.value},
            {"state", stage_name(journal.stage)},
            {"events", events_json(journal.events)},
            {"devices", cards},
            {"touched", touched}};
}
Json receipt_json(const Receipt& receipt) {
    Json cards = Json::array();
    for (const auto& entry : receipt.plan.devices)
        cards.push_back({{"role", entry.device.role.label()},
                         {"bdf", entry.device.pci.value},
                         {"settings", tuning_json(entry.tuning, false)}});
    Json observations = Json::object();
    for (const auto& item : receipt.observations) {
        Json measured_cards = Json::array();
        for (const auto& card : item.cards)
            measured_cards.push_back(observed_json(card));
        observations[moment_label(item.at)] = {
            {"status", observation_label(item.state)},
            {"reason", item.reason.empty() ? Json(nullptr) : Json(item.reason)},
            {"cards", std::move(measured_cards)}};
    }
    Json manifest = {
        {"status", "resolved"}, {"content_digest", receipt.plan.manifest_digest.value}, {"reason", nullptr}};
    Json request = {{"mode", receipt.plan.mode == Mode::apply ? "apply" : "observe"},
                    {"profile", receipt.plan.profile},
                    {"profile_digest", receipt.plan.profile_digest.value},
                    {"devices", cards}};
    const std::string_view receipt_state =
        receipt.stage == Stage::fresh ? "not_requested" : stage_name(receipt.stage);
    Json result = {{"schema_version", 1},
                   {"run_id", receipt.run_id},
                   {"manifest", manifest},
                   {"request", request},
                   {"observations", observations},
                   {"transaction", {{"state", receipt_state}, {"events", events_json(receipt.events)}}}};
    if (!receipt.workload.empty())
        result["workload"] = receipt.workload;
    return result;
}

v2::Result<Journal> parse_journal(const Json& raw, const fs::path& from) {
    try {
        if (!raw.is_object() || raw.at("schema_version") != 1 || !raw.at("devices").is_array() ||
            !raw.at("events").is_array() || !raw.at("touched").is_array())
            return error(Code::invalid_format, from.string(), "invalid journal schema");
        Journal result;
        result.run_id = raw.at("run_id").get<std::string>();
        if (result.run_id.empty() || result.run_id.size() > 128U ||
            from.filename().string() != result.run_id + ".json")
            return error(Code::invalid_format, from.string(), "invalid journal run ID or filename");
        for (char ch : result.run_id) {
            if (!(ch >= 'a' && ch <= 'z') && !(ch >= 'A' && ch <= 'Z') && !(ch >= '0' && ch <= '9') &&
                ch != '_' && ch != '-')
                return error(Code::invalid_format, from.string(), "unsafe journal run ID");
        }
        auto manifest = Digest::parse(raw.at("manifest_digest").get<std::string>());
        auto profile = Digest::parse(raw.at("profile_digest").get<std::string>());
        if (!manifest || !profile)
            return error(Code::invalid_format, from.string(), "invalid journal manifest/profile binding");
        result.manifest_digest = manifest.value();
        result.profile_digest = profile.value();
        auto stage = stage_from_name(raw.at("state").get<std::string>());
        if (!stage)
            return stage.error();
        result.stage = stage.value();
        for (const auto& entry : raw.at("devices")) {
            auto pci = PciAddress::parse(entry.at("bdf").get<std::string>());
            if (!pci)
                return pci.error();
            auto tuning = parse_tuning(entry.at("setting"));
            auto role = Role::parse(entry.at("role").get<std::string>());
            if (!tuning)
                return tuning.error();
            if (!role)
                return role.error();
            CardSnapshot before;
            before.pci = pci.value();
            before.amdgpu = true;
            if (!entry.at("hwmon").is_null()) {
                std::string full = entry.at("hwmon").get<std::string>();
                const auto prefix = "/sys/bus/pci/devices/" + pci.value().value + "/hwmon/";
                if (!full.starts_with(prefix))
                    return error(Code::invalid_format, from.string(), "invalid hardware monitor path");
                auto hw = full.substr(prefix.size());
                if (!hw.starts_with("hwmon") || hw.size() < 6U)
                    return error(Code::invalid_format, from.string(), "invalid hwmon name");
                for (char ch : hw.substr(5))
                    if (ch < '0' || ch > '9')
                        return error(Code::invalid_format, from.string(), "invalid hwmon numeric suffix");
                before.hwmon = std::move(hw);
            }
            if (auto failure = load_original(before, entry.at("before")))
                return *failure;
            if (before.at(ControlId::shader_offset).current) {
                before.identity = entry.value("identity", std::string());
                if (before.identity.empty())
                    return error(Code::invalid_format, from.string(), "signed shader journal lacks original identity");
            }
            result.devices.push_back(
                {DevicePolicy{{role.value(), pci.value()}, tuning.value()}, std::move(before)});
        }
        for (const auto& entry : raw.at("touched")) {
            auto pci = PciAddress::parse(entry.get<std::string>());
            if (!pci)
                return pci.error();
            result.touched.push_back(pci.value());
        }
        for (const auto& entry : raw.at("events")) {
            auto kind = parse_event(entry.at("kind").get<std::string>());
            if (!kind)
                return kind.error();
            std::optional<PciAddress> pci;
            if (entry.contains("bdf")) {
                auto parsed = PciAddress::parse(entry.at("bdf").get<std::string>());
                if (!parsed)
                    return parsed.error();
                pci = parsed.value();
            }
            result.events.push_back({kind.value(), pci, entry.value("reason", std::string())});
        }
        return result;
    } catch (const std::exception& e) {
        return error(Code::invalid_format, from.string(), std::string("invalid recovery journal: ") + e.what());
    }
}
bool valid_run_id(std::string_view id) {
    if (id.empty() || id.size() > 128U)
        return false;
    return std::all_of(id.begin(), id.end(), [](char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' ||
               ch == '_';
    });
}
} // namespace

fs::path DiskRecords::do_journal_path(std::string_view run_id) const {
    return journal_dir_ / (std::string(run_id) + ".json");
}
v2::Result<bool> DiskRecords::do_contains(std::string_view run_id) {
    if (!valid_run_id(run_id))
        return error(Code::invalid_value, "power.run_id", "invalid power run ID");
    std::error_code ec;
    const auto st = fs::symlink_status(journal_path(run_id), ec);
    if (ec == std::errc::no_such_file_or_directory)
        return false;
    if (ec)
        return error(Code::io_failure, "power.journal", ec.message());
    return st.type() != fs::file_type::not_found;
}
v2::Status DiskRecords::do_save(const Journal& journal) {
    if (!valid_run_id(journal.run_id))
        return error(Code::invalid_value, "power.run_id", "invalid power run ID");
    return durable_write(journal_path(journal.run_id), journal_json(journal));
}
v2::Status DiskRecords::do_save(const Receipt& receipt) {
    if (!valid_run_id(receipt.run_id))
        return error(Code::invalid_value, "power.run_id", "invalid power run ID");
    return durable_write(receipt_dir_ / (receipt.run_id + ".power.json"), receipt_json(receipt));
}
v2::Status DiskRecords::do_reconcile_recovery(const Journal& journal, const Observation& stop) {
    if (!valid_run_id(journal.run_id))
        return error(Code::invalid_value, "power.run_id", "invalid recovery receipt run ID");
    const auto path = receipt_dir_ / (journal.run_id + ".power.json");
    auto loaded = durable_read(path);
    if (!loaded)
        return loaded.error();
    auto& receipt = loaded.value();
    try {
        if (!receipt.is_object() || receipt.at("schema_version") != 1 ||
            receipt.at("run_id") != journal.run_id || receipt.at("manifest").at("status") != "resolved" ||
            receipt.at("manifest").at("content_digest") != journal.manifest_digest.value ||
            receipt.at("request").at("mode") != "apply" ||
            receipt.at("request").at("profile_digest") != journal.profile_digest.value ||
            !receipt.at("request").at("devices").is_array() ||
            receipt.at("request").at("devices").size() < journal.devices.size())
            return error(Code::invalid_format, path.string(), "recovery receipt identity mismatch");
        // The journal binds the manifest and profile digests; validate both
        // against the existing receipt and verify every selected role, BDF
        // and setting before resolving the recovery journal.
        for (const auto& saved : journal.devices) {
            bool matched = false;
            for (const auto& device : receipt.at("request").at("devices")) {
                if (device.at("bdf") == saved.policy.device.pci.value &&
                    device.at("role") == saved.policy.device.role.label() &&
                    device.at("settings") == tuning_json(saved.policy.tuning, false)) {
                    matched = true;
                    break;
                }
            }
            if (!matched)
                return error(Code::invalid_format, path.string(), "recovery receipt controls mismatch");
        }
        auto& transaction = receipt.at("transaction");
        if (!transaction.is_object() || !transaction.at("state").is_string())
            return error(Code::invalid_format, path.string(), "invalid receipt transaction");
        transaction["state"] = stage_name(journal.stage);
        transaction["events"] = events_json(journal.events);
        auto& observations = receipt.at("observations");
        if (!observations.is_object())
            return error(Code::invalid_format, path.string(), "invalid receipt observations");
        Json cards = Json::array();
        for (const auto& card : stop.cards)
            cards.push_back(observed_json(card));
        observations["stop"] = {{"status", observation_label(stop.state)},
                                {"reason", stop.reason.empty() ? Json(nullptr) : Json(stop.reason)},
                                {"cards", std::move(cards)}};
        return durable_write(path, receipt);
    } catch (const std::exception& failure) {
        return error(Code::invalid_format, path.string(),
                     std::string("invalid recovery receipt: ") + failure.what());
    }
}

v2::Result<Journal> DiskRecords::do_load(const fs::path& path) {
    if (path.parent_path().lexically_normal() != journal_dir_.lexically_normal() || path.extension() != ".json")
        return error(Code::invalid_value, path.string(),
                     "power journal must be inside owned journal directory");
    auto data = durable_read(path);
    if (!data)
        return data.error();
    return parse_journal(data.value(), path);
}
v2::Result<std::vector<fs::path>> DiskRecords::do_unresolved_journals() {
    std::error_code ec;
    if (!fs::exists(journal_dir_, ec)) {
        if (ec)
            return error(Code::io_failure, journal_dir_.string(), ec.message());
        return std::vector<fs::path>{};
    }
    auto dir = directory(journal_dir_, false);
    if (!dir)
        return dir.error();
    const int scanning_fd = ::dup(dir.value().get());
    if (scanning_fd < 0)
        return error(Code::io_failure, journal_dir_.string(), "cannot duplicate journal fd");
    struct DirectoryCloser {
        void operator()(DIR* dir) const noexcept {
            if (dir)
                (void)::closedir(dir);
        }
    };
    std::unique_ptr<DIR, DirectoryCloser> stream(::fdopendir(scanning_fd));
    if (!stream) {
        (void)::close(scanning_fd);
        return error(Code::io_failure, journal_dir_.string(), "cannot enumerate journal directory");
    }
    std::vector<fs::path> pending;
    for (;;) {
        errno = 0;
        dirent* entry = ::readdir(stream.get());
        if (!entry) {
            if (errno != 0)
                return error(Code::io_failure, journal_dir_.string(), "journal directory iteration failed");
            break;
        }
        std::string filename(entry->d_name);
        if (!filename.ends_with(".json"))
            continue;
        const auto path = journal_dir_ / filename;
        auto record = load(path);
        if (!record)
            return error(Code::invalid_format, path.string(),
                         "unreadable power journal blocks all power writes: " + record.error().message);
        if (unresolved(record.value().stage))
            pending.push_back(path);
    }
    std::sort(pending.begin(), pending.end());
    return pending;
}

} // namespace bc::serving::power
