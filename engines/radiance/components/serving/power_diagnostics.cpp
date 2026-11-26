#include "power_transaction.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <thread>

#include <sys/utsname.h>

namespace bc::serving::power {
namespace {
using Code = v2::ConfigErrorCode;
v2::ConfigError bad(std::string path, std::string reason) {
    return v2::make_error(Code::invalid_value, std::move(path), std::move(reason));
}
std::optional<std::string> read(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream)
        return {};
    std::stringstream text;
    text << stream.rdbuf();
    if (stream.bad())
        return {};
    auto output = text.str();
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r' || output.back() == ' '))
        output.pop_back();
    return output;
}
std::optional<int64_t> integer(const fs::path& path) {
    auto text = read(path);
    if (!text)
        return {};
    int64_t value = 0;
    const auto [where, ec] = std::from_chars(text->data(), text->data() + text->size(), value);
    if (ec != std::errc{} || where != text->data() + text->size())
        return {};
    return value;
}
struct Extrema {
    int64_t low = std::numeric_limits<int64_t>::max();
    int64_t high = std::numeric_limits<int64_t>::min();
    void push(int64_t number) {
        low = std::min(low, number);
        high = std::max(high, number);
    }
    bool operator==(const Extrema&) const = default;
};
std::array<Extrema, 3> spread(const std::vector<Sample>& values) {
    std::array<Extrema, 3> extents{};
    for (const auto& value : values) {
        extents[0].push(value.power.value);
        extents[1].push(value.voltage.value);
        extents[2].push(value.busy_percent);
    }
    return extents;
}
v2::Result<Window> unavailable(Window window, std::string reason) {
    window.reason = std::move(reason);
    window.acquisition = CaptureState::unavailable;
    return window;
}
} // namespace

Verdict evaluate(const std::optional<Window>& floor, const std::optional<Window>& idle) {
    if (!floor && !idle)
        return {GfxoffState::unavailable, GfxoffReason::incomplete_window};
    const Window& available = idle ? *idle : *floor;
    if (available.acquisition == CaptureState::not_applicable)
        return {GfxoffState::not_applicable, GfxoffReason::not_gfx12};
    if (!floor || !idle)
        return {GfxoffState::needs_idle_context, GfxoffReason::no_engine_context};
    if (floor->acquisition != CaptureState::measured || idle->acquisition != CaptureState::measured)
        return {GfxoffState::unavailable, GfxoffReason::failed_capture};
    if (floor->phase != WindowPhase::no_process || idle->phase != WindowPhase::engine_loaded_idle ||
        floor->pci != idle->pci || floor->identity != idle->identity ||
        floor->sensor_identity != idle->sensor_identity)
        return {GfxoffState::ambiguous, GfxoffReason::identity_mismatch};
    if (floor->samples.size() < 2U || idle->samples.size() < 2U)
        return {GfxoffState::unavailable, GfxoffReason::insufficient_samples};
    if (std::any_of(floor->samples.begin(), floor->samples.end(),
                    [](const Sample& sample) { return sample.busy_percent != 0; }))
        return {GfxoffState::ambiguous, GfxoffReason::floor_not_idle};
    const auto latest_floor =
        std::max_element(floor->samples.begin(), floor->samples.end(),
                         [](const Sample& a, const Sample& b) { return a.timestamp < b.timestamp; });
    const auto first_idle =
        std::min_element(idle->samples.begin(), idle->samples.end(),
                         [](const Sample& a, const Sample& b) { return a.timestamp < b.timestamp; });
    if (latest_floor->timestamp >= first_idle->timestamp)
        return {GfxoffState::ambiguous, GfxoffReason::time_order_mismatch};
    const auto low = spread(floor->samples), high = spread(idle->samples);
    bool elevated = true, same = true;
    for (size_t index = 0; index < low.size(); ++index) {
        elevated &= low[index].high < high[index].low;
        same &= low[index] == high[index];
    }
    if (elevated)
        return {GfxoffState::absent, GfxoffReason::elevated_signals};
    if (same)
        return {GfxoffState::active, GfxoffReason::matching_floor};
    return {GfxoffState::ambiguous, GfxoffReason::mixed_signals};
}

namespace {
std::string_view reason_label(GfxoffReason reason) {
    switch (reason) {
    case GfxoffReason::matching_floor:
        return "loaded_idle_matches_measured_floor_spread";
    case GfxoffReason::elevated_signals:
        return "loaded_idle_power_voltage_and_busy_above_floor";
    case GfxoffReason::incomplete_window:
        return "missing_both_windows";
    case GfxoffReason::no_engine_context:
        return "paired_no_process_and_loaded_idle_windows_required";
    case GfxoffReason::not_gfx12:
        return "non_gfx12_card";
    case GfxoffReason::failed_capture:
        return "measurement_unavailable";
    case GfxoffReason::identity_mismatch:
        return "phase_card_sensor_or_driver_identity_mismatch";
    case GfxoffReason::floor_not_idle:
        return "no_process_floor_not_idle";
    case GfxoffReason::time_order_mismatch:
        return "floor_not_before_loaded_idle";
    case GfxoffReason::insufficient_samples:
        return "insufficient_samples";
    case GfxoffReason::mixed_signals:
        return "noise_or_mixed_signals_prevent_classification";
    }
    return "invalid_verdict";
}
ObservationState observation_state(GfxoffState state) {
    switch (state) {
    case GfxoffState::active:
        return ObservationState::gfxoff_active;
    case GfxoffState::absent:
        return ObservationState::gfxoff_absent;
    case GfxoffState::ambiguous:
        return ObservationState::gfxoff_ambiguous;
    case GfxoffState::unavailable:
        return ObservationState::unavailable;
    case GfxoffState::not_applicable:
        return ObservationState::gfxoff_not_applicable;
    case GfxoffState::needs_idle_context:
        return ObservationState::gfxoff_needs_context;
    }
    return ObservationState::unavailable;
}
} // namespace

v2::Status SessionPower::report_gfxoff(const Verdict& verdict) {
    if (journal_.stage == Stage::fresh)
        return v2::make_error(v2::ConfigErrorCode::invalid_state, "power.gfxoff",
                              "cannot report GFXOFF before session begins");
    receipt_.observations.push_back(
        {Moment::gfxoff, observation_state(verdict.state), std::string(reason_label(verdict.reason)), {}});
    return records_.save(receipt_);
}

LinuxSensor::LinuxSensor(fs::path sys_root, Interval between)
    : sys_root_(std::move(sys_root)), between_(std::move(between)) {
    if (!between_)
        between_ = [] { std::this_thread::sleep_for(std::chrono::seconds(1)); };
}

v2::Result<Window> LinuxSensor::do_measure(const PciAddress& pci, WindowPhase phase, size_t samples) {
    if (!PciAddress::parse(pci.value) || samples < 2U || samples > 1024U)
        return bad("power.gfxoff", "invalid PCI BDF or sample count");
    Window window;
    window.pci = pci;
    window.phase = phase;
    const fs::path device = sys_root_ / "bus/pci/devices" / pci.value;
    const fs::path die = device / "ip_discovery/die";
    std::set<int64_t> gc_majors;
    std::error_code ec;
    if (fs::is_directory(die, ec)) {
        for (const auto& chip : fs::directory_iterator(die)) {
            auto gc = chip.path() / "GC";
            if (!fs::is_directory(gc, ec))
                continue;
            for (const auto& unit : fs::directory_iterator(gc)) {
                if (auto major = integer(unit.path() / "major"))
                    gc_majors.insert(*major);
            }
        }
    }
    if (gc_majors.size() != 1U)
        return unavailable(std::move(window), "GC IP discovery is unavailable or ambiguous");
    if (*gc_majors.begin() != 12) {
        window.acquisition = CaptureState::not_applicable;
        window.reason = "card is not gfx12";
        return window;
    }
    struct utsname platform{};
    if (::uname(&platform) != 0)
        return unavailable(std::move(window), "kernel identity not readable");
    auto pci_device = read(device / "device");
    auto amdgpu_srcversion = read(sys_root_ / "module/amdgpu/srcversion");
    if (!pci_device || !amdgpu_srcversion)
        return unavailable(std::move(window), "PCI/driver identity not readable");
    window.identity = std::string(platform.release) + ":" + *pci_device + ":" + *amdgpu_srcversion;
    const fs::path hwmon = device / "hwmon";
    std::vector<fs::path> monitors;
    if (fs::is_directory(hwmon, ec)) {
        for (const auto& entry : fs::directory_iterator(hwmon)) {
            const auto sensor = entry.path() / "name";
            if (auto name = read(sensor); name && *name == "amdgpu")
                monitors.push_back(entry.path());
        }
    }
    if (monitors.size() != 1U)
        return unavailable(std::move(window), "amdgpu hwmon missing or ambiguous");
    const auto power = monitors.front() / "power1_average";
    const auto voltage = monitors.front() / "in0_input";
    const auto busy = device / "gpu_busy_percent";
    window.sensor_identity = power.string() + ";" + voltage.string() + ";" + busy.string();
    for (size_t sample = 0; sample < samples; ++sample) {
        if (sample != 0U) {
            if (auto failure = between_.invoke("power.gfxoff.interval"))
                return *failure;
        }
        const auto p = integer(power), v = integer(voltage), b = integer(busy);
        if (!p || !v || !b || *p <= 0 || *v < 0 || *b < 0 || *b > 100)
            return unavailable(std::move(window), "power, voltage or busy sensor unreadable");
        window.samples.push_back(
            {Microwatts{*p}, Millivolts{*v}, static_cast<uint8_t>(*b), std::chrono::system_clock::now()});
    }
    window.acquisition = CaptureState::measured;
    return window;
}

std::vector<HostFinding> inspect_host(const PowerPlan& plan, Hardware& hardware,
                                      std::span<const fs::path> rapl_energy,
                                      const callback::Guarded<bool(const fs::path&)>& readable) {
    std::vector<HostFinding> found;
    if (plan.devices.empty())
        found.push_back({HostIssue::invalid_selection, {}, "empty GPU selection"});
    std::set<std::string> seen;
    for (const auto& device : plan.devices) {
        const auto& pci = device.device.pci;
        if (!PciAddress::parse(pci.value) || !seen.insert(pci.value).second) {
            found.push_back({HostIssue::invalid_selection, pci, "invalid/duplicate selected BDF"});
            continue;
        }
        if (!device.tuning.changes_hardware())
            continue;
        const auto snapshot = hardware.inspect(pci);
        if (!snapshot) {
            found.push_back({HostIssue::probe_failed, pci, snapshot.error().message});
            continue;
        }
        if (!snapshot.value().amdgpu) {
            found.push_back({HostIssue::missing_backend, pci, "GPU has no amdgpu power controller"});
            continue;
        }
        const auto inspect = [&](ControlId id, int64_t required) {
            const auto& control = snapshot.value().at(id);
            if (!control.writable || !control.current || !control.minimum || !control.maximum)
                found.push_back({HostIssue::missing_control, pci, "required GPU control is not writable"});
            else if (required < *control.minimum || required > *control.maximum)
                found.push_back({HostIssue::unsafe_range, pci, "requested control outside device range"});
        };
        const auto& setting = device.tuning;
        if (setting.voltage_offset)
            inspect(ControlId::voltage, setting.voltage_offset->value);
        if (setting.shader_ceiling)
            inspect(ControlId::shader, setting.shader_ceiling->value);
        if (setting.shader_offset)
            inspect(ControlId::shader_offset, setting.shader_offset->value);
        if (setting.memory_ceiling)
            inspect(ControlId::memory, setting.memory_ceiling->value);
        if (setting.board_cap) {
            if (setting.board_cap->value > 9007199254LL || setting.board_cap->value <= 0)
                found.push_back({HostIssue::unsafe_range, pci, "board cap cannot be converted safely"});
            else
                inspect(ControlId::board_cap, setting.board_cap->value * 1000000LL);
        }
    }
    for (const auto& path : rapl_energy) {
        const auto verified = readable.invoke("power.host.rapl", path);
        if (!verified || !verified.value()) {
            found.push_back(
                {HostIssue::cpu_energy_unreadable,
                 {},
                 (verified ? "CPU package energy is not readable at " : "CPU package energy probe failed at ") +
                     path.string() + (verified ? "" : ": " + verified.error().message)});
            break;
        }
    }
    return found;
}

} // namespace bc::serving::power
