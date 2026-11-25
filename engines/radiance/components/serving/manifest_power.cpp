#include "manifest_power.h"
#include "manifest_writer.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>
#include <unistd.h>

namespace bc::serving::session {
namespace {
using Code = v2::ConfigErrorCode;
v2::ConfigError refusal(std::string reason) {
    return v2::make_error(Code::admission_refused, "session.power", std::move(reason));
}
v2::Result<std::vector<power::SelectedGpu>> selection(const v2::Manifest& manifest) {
    std::vector<power::SelectedGpu> result;
    std::set<std::string> unique;
    for (const auto& device : manifest.spec.processes.engine.device.gpus) {
        auto pci = power::PciAddress::parse(device);
        if (!pci)
            return pci.error();
        if (!unique.insert(pci.value().value).second)
            return refusal("engine and helpers must have distinct PCI addresses");
        result.push_back({power::Role{}, std::move(pci.value())});
    }
    if (result.empty())
        return refusal("no engine PCI addresses");
    for (size_t index = 0; index < manifest.spec.processes.helpers.size(); ++index) {
        const auto& helper = manifest.spec.processes.helpers[index];
        const auto expected = "helper" + std::to_string(index);
        if (helper.role != expected)
            return refusal("helper role is not in manifest order: " + helper.role);
        auto role = power::Role::parse(helper.role);
        auto pci = power::PciAddress::parse(helper.gpu);
        if (!role)
            return role.error();
        if (!pci)
            return pci.error();
        if (!unique.insert(pci.value().value).second)
            return refusal("engine and helpers must have distinct PCI addresses");
        result.push_back({role.value(), std::move(pci.value())});
    }
    return result;
}
bool same_policy(const power::PowerPlan& x, const power::PowerPlan& y) {
    if (x.mode != y.mode || x.profile != y.profile || x.profile_digest.value != y.profile_digest.value ||
        x.devices.size() != y.devices.size())
        return false;
    for (size_t i = 0; i < x.devices.size(); ++i) {
        const auto& a = x.devices[i];
        const auto& b = y.devices[i];
        if (!(a.device.role == b.device.role) || !(a.device.pci == b.device.pci) ||
            a.tuning.voltage_offset != b.tuning.voltage_offset ||
            a.tuning.shader_ceiling != b.tuning.shader_ceiling ||
            a.tuning.memory_ceiling != b.tuning.memory_ceiling || a.tuning.board_cap != b.tuning.board_cap)
            return false;
    }
    return true;
}
std::string file_text(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in)
        return {};
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
std::optional<pid_t> parent_of(pid_t pid, const std::filesystem::path& proc) {
    std::ifstream in(proc / std::to_string(pid) / "stat");
    std::string line;
    if (!std::getline(in, line))
        return {};
    const size_t end = line.rfind(") ");
    if (end == std::string::npos)
        return {};
    std::istringstream rest(line.substr(end + 2));
    char state = 0;
    pid_t parent = 0;
    if (!(rest >> state >> parent) || state == 'Z' || state == 'X')
        return {};
    return parent;
}
bool unexpired(std::string_view input) {
    // The lease timestamp must be a complete UTC RFC3339 instant. Never
    // accept a numeric prefix followed by arbitrary text or a local offset.
    if (input.size() < 20 || input[4] != '-' || input[7] != '-' || input[10] != 'T' || input[13] != ':' ||
        input[16] != ':')
        return false;
    const auto number = [input](size_t begin, size_t width) -> std::optional<int> {
        int value = 0;
        const auto* first = input.data() + begin;
        const auto parsed = std::from_chars(first, first + width, value);
        if (parsed.ec != std::errc{} || parsed.ptr != first + width)
            return std::nullopt;
        return value;
    };
    const auto year = number(0, 4), month = number(5, 2), day = number(8, 2);
    const auto hour = number(11, 2), minute = number(14, 2), second = number(17, 2);
    if (!year || !month || !day || !hour || !minute || !second)
        return false;
    size_t zone = 19;
    if (input[zone] == '.') {
        ++zone;
        const size_t start = zone;
        while (zone < input.size() && input[zone] >= '0' && input[zone] <= '9')
            ++zone;
        if (zone == start)
            return false;
    }
    if (input.substr(zone) != "Z" && input.substr(zone) != "+00:00")
        return false;
    if (*year < 2020 || *month < 1 || *month > 12 || *day < 1 || *day > 31 || *hour < 0 || *hour > 23 ||
        *minute < 0 || *minute > 59 || *second < 0 || *second > 59)
        return false;
    std::tm utc{};
    utc.tm_year = *year - 1900;
    utc.tm_mon = *month - 1;
    utc.tm_mday = *day;
    utc.tm_hour = *hour;
    utc.tm_min = *minute;
    utc.tm_sec = *second;
    const std::tm expected = utc;
    const auto expires = ::timegm(&utc);
    // timegm normalizes impossible dates; reject those before trusting expiry.
    return utc.tm_year == expected.tm_year && utc.tm_mon == expected.tm_mon &&
           utc.tm_mday == expected.tm_mday && utc.tm_hour == expected.tm_hour &&
           utc.tm_min == expected.tm_min && utc.tm_sec == expected.tm_sec &&
           expires > std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
}

} // namespace

bool owner_has_gpu_lease(const std::filesystem::path& leases, const std::filesystem::path& proc) {
    const auto bytes = file_text(leases / "gpu.json");
    if (bytes.empty())
        return false;
    const auto data = nlohmann::json::parse(bytes, nullptr, false);
    if (!data.is_object() || !data.contains("expires") || !data["expires"].is_string() ||
        !data.contains("pids") || !data["pids"].is_array() || !unexpired(data["expires"].get<std::string>()))
        return false;
    const char* instance = std::getenv("AGENT_INSTANCE");
    if (instance == nullptr || *instance == '\0' || !data.contains("instance") ||
        !data["instance"].is_string() || data["instance"] != instance)
        return false; // apply needs an explicit verified instance, not merely matching PIDs
    std::set<pid_t> live;
    for (const auto& pid : data["pids"]) {
        if (!pid.is_number_integer() || pid.get<int64_t>() <= 1 ||
            pid.get<int64_t>() > std::numeric_limits<pid_t>::max())
            return false;
        auto candidate = static_cast<pid_t>(pid.get<int64_t>());
        if (parent_of(candidate, proc))
            live.insert(candidate);
    }
    if (live.empty())
        return false;
    pid_t current = ::getpid();
    for (size_t step = 0; step < 64U && current > 1; ++step) {
        if (live.contains(current))
            return true;
        const auto parent = parent_of(current, proc);
        if (!parent || *parent == current)
            break;
        current = *parent;
    }
    return false;
}

ManifestPower::ManifestPower(power::Hardware& hardware, power::Records& records, power::Sensor& sensor,
                             power::SessionPower::Predicate lease, bool allow_apply, std::string workload)
    : hardware_(hardware), records_(records), sensor_(sensor), lease_(std::move(lease)),
      allow_apply_(allow_apply), workload_(std::move(workload)) {}

v2::Status ManifestPower::begin(const v2::Manifest& manifest, v2::PlanUse use) {
    if (transaction_)
        return refusal("power was already bound to a session manifest");
    (void)use;
    const auto selected = selection(manifest);
    if (!selected)
        return selected.error();
    const auto plan = power::bind(manifest, selected.value());
    if (!plan)
        return plan.error();
    if (plan.value().mode == power::Mode::apply && !allow_apply_)
        return refusal("hardware power apply requires explicit --allow-power-apply");
    for (const auto& bdf : manifest.spec.processes.engine.device.gpus) {
        const auto pci = power::PciAddress::parse(bdf);
        if (!pci)
            return pci.error();
        engine_.push_back(pci.value());
    }
    bound_ = plan.value();
    // Each owner execution owns exactly one durable power transaction and
    // receipt. The authenticated instance id names the immutable origin only.
    execution_ = PowerExecution{"power-" + new_instance_id(), manifest.metadata.instance_id};
    transaction_ = std::make_unique<power::SessionPower>(plan.value(), execution_->run_id, hardware_, records_,
                                                         lease_, [this] { return verified_exit_; });
    transaction_->set_workload(workload_);
    return transaction_->begin();
}

v2::Status ManifestPower::verify_serving(const std::filesystem::path& manifest) {
    if (!transaction_ || !bound_)
        return refusal("serving manifest appeared before authenticated power admission");
    const auto authenticated = v2::load_manifest(manifest, v2::PlanUse::serving);
    if (!authenticated)
        return authenticated.error();
    const auto selected = selection(authenticated.value());
    if (!selected)
        return selected.error();
    const auto plan = power::bind(authenticated.value(), selected.value());
    if (!plan)
        return plan.error();
    if (!same_policy(*bound_, plan.value()))
        return refusal("measured serving manifest changed admitted power profile, settings or ordered BDFs");
    return std::nullopt;
}

v2::Status ManifestPower::capture(power::WindowPhase phase) {
    if (!transaction_)
        return refusal("GFXOFF capture requested before power admission");
    if (phase == power::WindowPhase::no_process) {
        floor_.clear();
        for (const auto& pci : engine_) {
            const auto window = sensor_.measure(pci, phase, 3U);
            if (window)
                floor_.push_back(window.value());
            else
                floor_.push_back(std::nullopt); // advisory sensor failure, never substitute a healthy floor
        }
        return std::nullopt;
    }
    for (size_t i = 0; i < engine_.size(); ++i) {
        const auto loaded = sensor_.measure(engine_[i], phase, 3U);
        const std::optional<power::Window> idle =
            loaded ? std::optional<power::Window>{loaded.value()} : std::nullopt;
        const auto previous = i < floor_.size() ? floor_[i] : std::optional<power::Window>{};
        if (auto error = transaction_->report_gfxoff(power::evaluate(previous, idle)))
            return error; // receipt persistence is mandatory, unlike unavailable sensor data
    }
    return std::nullopt;
}

void ManifestPower::children_verified() noexcept {
    verified_exit_ = true;
}
v2::Status ManifestPower::finish() {
    if (!transaction_)
        return refusal("power restoration requested without a transaction");
    return transaction_->finish();
}
} // namespace bc::serving::session
