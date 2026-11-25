#include "power_io.h"
#include "power_transaction.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <fstream>
#include <sstream>
#include <string_view>
#include <system_error>
#include <type_traits>

#include <fcntl.h>
#include <unistd.h>

namespace bc::serving::power {
namespace {
using Code = v2::ConfigErrorCode;
v2::ConfigError problem(Code code, const fs::path& where, std::string message) {
    return v2::make_error(code, where.string(), std::move(message));
}
std::string trim(std::string_view input) {
    const auto left = input.find_first_not_of(" \n\t\r");
    if (left == std::string_view::npos)
        return {};
    return std::string(input.substr(left, input.find_last_not_of(" \n\t\r") + 1U - left));
}
std::optional<int64_t> parse_number(std::string_view text, std::string_view suffix = {}) {
    const auto cleaned = trim(text);
    std::string_view number(cleaned);
    if (!suffix.empty()) {
        if (number.size() < suffix.size())
            return std::nullopt;
        auto end = number.substr(number.size() - suffix.size());
        for (size_t i = 0; i < suffix.size(); ++i) {
            if (std::tolower(static_cast<unsigned char>(end[i])) !=
                std::tolower(static_cast<unsigned char>(suffix[i])))
                return std::nullopt;
        }
        number.remove_suffix(suffix.size());
        const auto number_text = trim(number);
        // The temporary must survive from_chars; parse here instead.
        int64_t value = 0;
        auto [ptr, ec] = std::from_chars(number_text.data(), number_text.data() + number_text.size(), value);
        if (ec != std::errc{} || ptr != number_text.data() + number_text.size())
            return std::nullopt;
        return value;
    }
    int64_t value = 0;
    auto [ptr, ec] = std::from_chars(number.data(), number.data() + number.size(), value);
    if (ec != std::errc{} || ptr != number.data() + number.size())
        return std::nullopt;
    return value;
}
std::optional<std::string> read_text(const fs::path& path) {
    std::ifstream file(path);
    if (!file)
        return {};
    std::ostringstream buffer;
    buffer << file.rdbuf();
    if (!file.good() && !file.eof())
        return {};
    return buffer.str();
}
std::optional<int64_t> read_number(const fs::path& path) {
    if (const auto contents = read_text(path))
        return parse_number(*contents);
    return {};
}
bool writable(const fs::path& path) {
    return ::access(path.c_str(), W_OK) == 0;
}
struct Overdrive {
    std::optional<int64_t> shader, memory, voltage;
    std::optional<std::pair<int64_t, int64_t>> shader_range, memory_range, voltage_range;
};
std::optional<std::pair<int64_t, int64_t>> clock_range(std::string_view row, std::string_view label,
                                                       std::string_view suffix) {
    if (!row.starts_with(label) || row.size() <= label.size() || row[label.size()] != ':')
        return {};
    std::istringstream words(trim(row.substr(label.size() + 1U)));
    std::string low, high, extra;
    if (!(words >> low >> high) || words >> extra)
        return {};
    auto a = parse_number(low, suffix), b = parse_number(high, suffix);
    if (!a || !b || *a > *b)
        return {};
    return std::pair{*a, *b};
}
Overdrive parse_overdrive(std::string_view contents) {
    Overdrive result;
    enum class Section : uint8_t { none, shader, memory, voltage, ranges };
    Section part = Section::none;
    bool shader_start = false, memory_start = false;
    std::istringstream input{std::string(contents)};
    std::string line;
    while (std::getline(input, line)) {
        auto row = trim(line);
        if (row == "OD_SCLK:") {
            part = Section::shader;
            continue;
        }
        if (row == "OD_MCLK:") {
            part = Section::memory;
            continue;
        }
        if (row == "OD_VDDGFX_OFFSET:") {
            part = Section::voltage;
            continue;
        }
        if (row == "OD_RANGE:") {
            part = Section::ranges;
            continue;
        }
        if (row.empty())
            continue;
        switch (part) {
        case Section::shader:
        case Section::memory: {
            const bool begin = row.starts_with("0:");
            const bool end = row.starts_with("1:");
            if (!begin && !end) {
                part = Section::none;
                break;
            }
            const auto parsed = parse_number(row.substr(2), "Mhz");
            if (part == Section::shader) {
                if (begin && parsed)
                    shader_start = true;
                if (end && parsed && shader_start)
                    result.shader = parsed;
            } else {
                if (begin && parsed)
                    memory_start = true;
                if (end && parsed && memory_start)
                    result.memory = parsed;
            }
            break;
        }
        case Section::voltage:
            result.voltage = parse_number(row, "mV");
            part = Section::none;
            break;
        case Section::ranges:
            if (auto value = clock_range(row, "SCLK", "Mhz"))
                result.shader_range = *value;
            else if (auto memory_limits = clock_range(row, "MCLK", "Mhz"))
                result.memory_range = *memory_limits;
            else if (auto voltage_limits = clock_range(row, "VDDGFX_OFFSET", "mV"))
                result.voltage_range = *voltage_limits;
            break;
        case Section::none:
            break;
        }
    }
    return result;
}
void assign(Capability& output, const std::optional<int64_t>& current,
            const std::optional<std::pair<int64_t, int64_t>>& range, bool can_write) {
    output.current = current;
    if (range) {
        output.minimum = range->first;
        output.maximum = range->second;
    }
    output.writable = bool(current && range && can_write);
}
bool valid_hwmon(std::string_view value) {
    if (!value.starts_with("hwmon") || value.size() <= 5U)
        return false;
    return std::all_of(value.begin() + 5, value.end(), [](char ch) { return ch >= '0' && ch <= '9'; });
}
} // namespace

v2::Result<SysfsWrite> render(const PowerCommand& command) {
    if (!PciAddress::parse(command.pci.value))
        return problem(Code::invalid_value, "power.command", "invalid PCI address");
    const fs::path root = fs::path("/sys/bus/pci/devices") / command.pci.value;
    return std::visit(
        [&](const auto& value) -> v2::Result<SysfsWrite> {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, BoardCap>) {
                if (!command.hwmon || !valid_hwmon(*command.hwmon))
                    return problem(Code::invalid_value, root, "invalid power monitor");
                if (value.value.value <= 0)
                    return problem(Code::invalid_value, root, "nonpositive board cap");
                return SysfsWrite{root / "hwmon" / *command.hwmon / "power1_cap",
                                  std::to_string(value.value.value)};
            } else {
                const auto path = root / "pp_od_clk_voltage";
                if constexpr (std::is_same_v<T, ShaderClock>)
                    return SysfsWrite{path, "s 1 " + std::to_string(value.value.value)};
                if constexpr (std::is_same_v<T, MemoryClock>)
                    return SysfsWrite{path, "m 1 " + std::to_string(value.value.value)};
                if constexpr (std::is_same_v<T, VoltageOffset>)
                    return SysfsWrite{path, "vo " + std::to_string(value.value.value)};
                return SysfsWrite{path, "c"};
            }
        },
        command.value);
}

v2::Result<CardSnapshot> LinuxHardware::do_inspect(const PciAddress& pci) {
    if (!PciAddress::parse(pci.value))
        return problem(Code::invalid_value, "power.device", "invalid PCI BDF");
    const auto root = root_ / "sys/bus/pci/devices" / pci.value;
    CardSnapshot snapshot;
    snapshot.pci = pci;
    std::error_code ec;
    const auto driver = fs::read_symlink(root / "driver", ec);
    snapshot.amdgpu = !ec && driver.filename() == "amdgpu";
    if (!snapshot.amdgpu)
        return snapshot;
    // Plain concatenation, compared by equality; any piece the host does not expose is simply empty.
    const auto piece = [](const fs::path& file) { return trim(read_text(file).value_or("")); };
    snapshot.identity = "kernel=" + piece(root_ / "proc/sys/kernel/osrelease") +
                        ";amdgpu=" + piece(root_ / "sys/module/amdgpu/version") + ";board=" + piece(root / "device") +
                        ":" + piece(root / "subsystem_vendor") + ":" + piece(root / "subsystem_device");
    const auto overdrive_file = root / "pp_od_clk_voltage";
    const auto od = read_text(overdrive_file);
    if (od) {
        const auto parsed = parse_overdrive(*od);
        const bool can_write = writable(overdrive_file);
        assign(snapshot.controls[static_cast<size_t>(ControlId::voltage)], parsed.voltage, parsed.voltage_range,
               can_write);
        assign(snapshot.controls[static_cast<size_t>(ControlId::shader)], parsed.shader, parsed.shader_range,
               can_write);
        assign(snapshot.controls[static_cast<size_t>(ControlId::memory)], parsed.memory, parsed.memory_range,
               can_write);
    }
    const fs::path base = root / "hwmon";
    std::vector<fs::path> monitors;
    const bool has_monitors = fs::is_directory(base, ec);
    if (ec && ec != std::errc::no_such_file_or_directory)
        return problem(Code::io_failure, base, "hwmon directory probe failed: " + ec.message());
    if (has_monitors) {
        fs::directory_iterator entry(base, ec);
        if (ec)
            return problem(Code::io_failure, base, "hwmon enumeration failed: " + ec.message());
        const fs::directory_iterator end;
        while (entry != end) {
            const auto path = entry->path();
            const auto name = path.filename().string();
            if (valid_hwmon(name)) {
                const bool is_mon = entry->is_directory(ec);
                if (ec)
                    return problem(Code::io_failure, path, "hwmon entry probe failed: " + ec.message());
                if (is_mon) {
                    const auto contents = read_text(path / "name");
                    if (contents && trim(*contents) == "amdgpu")
                        monitors.push_back(path);
                }
            }
            entry.increment(ec);
            if (ec)
                return problem(Code::io_failure, base, "hwmon enumeration failed: " + ec.message());
        }
    }
    if (monitors.size() == 1U) {
        snapshot.hwmon = monitors.front().filename().string();
        Capability& cap = snapshot.controls[static_cast<size_t>(ControlId::board_cap)];
        cap.current = read_number(monitors.front() / "power1_cap");
        cap.minimum = read_number(monitors.front() / "power1_cap_min");
        cap.maximum = read_number(monitors.front() / "power1_cap_max");
        cap.writable = cap.current && cap.minimum && cap.maximum && *cap.minimum <= *cap.maximum &&
                       writable(monitors.front() / "power1_cap");
    }
    return snapshot;
}
v2::Status LinuxHardware::do_execute(const PowerCommand& command) {
    auto operation = render(command);
    if (!operation)
        return operation.error();
    const auto target = root_ / operation.value().path.relative_path();
    detail::FileDescriptor fd(::open(target.c_str(), O_WRONLY | O_CLOEXEC | O_NOFOLLOW));
    if (!fd)
        return problem(Code::io_failure, target, "cannot open power sysfs control");
    const std::string contents = operation.value().value + "\n";
    size_t written = 0;
    while (written < contents.size()) {
        const auto n = ::write(fd.get(), contents.data() + written, contents.size() - written);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            return problem(Code::io_failure, target, "power sysfs write failed");
        written += static_cast<size_t>(n);
    }
    return std::nullopt;
}

} // namespace bc::serving::power
