#include "v2/p1_live_room.h"
#include "v2/error.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <system_error>

namespace bc::serving::v2 {
namespace {
constexpr int64_t kMib = 1024 * 1024;
constexpr int64_t kMax = 9007199254740991LL;
using Json = nlohmann::json;

ConfigError refuse(std::string path, std::string reason) {
  return make_error(ConfigErrorCode::invalid_value,
                    "measurement.host." + std::move(path), std::move(reason),
                    "remeasure live host, cgroup and AMD pin headroom");
}

bool valid_bdf(const std::string &bdf) {
  if (bdf.size() != 12 || bdf[4] != ':' || bdf[7] != ':' || bdf[10] != '.')
    return false;
  for (size_t i = 0; i < bdf.size(); ++i) {
    if (i == 4 || i == 7 || i == 10)
      continue;
    if (i == 11) {
      if (bdf[i] < '0' || bdf[i] > '7')
        return false;
    } else if (!std::isxdigit(static_cast<unsigned char>(bdf[i])) ||
               std::isupper(static_cast<unsigned char>(bdf[i])))
      return false;
  }
  return true;
}

bool mib(const Json &value, int64_t *out, bool positive = true) {
  if (!value.is_number_integer() && !value.is_number_unsigned())
    return false;
  const int64_t limit = kMax / kMib;
  if (value.is_number_unsigned()) {
    const auto v = value.get<uint64_t>();
    if (v > uint64_t(limit) || (positive && v == 0))
      return false;
    *out = int64_t(v) * kMib;
    return true;
  }
  const int64_t v = value.get<int64_t>();
  if (v < (positive ? 1 : 0) || v > limit)
    return false;
  *out = v * kMib;
  return true;
}

bool bytes_from_file(const std::filesystem::path &path, int64_t *out) {
  std::ifstream in(path);
  std::string token, extra;
  if (!in || !(in >> token) || (in >> extra) || token.empty())
    return false;
  uint64_t n = 0;
  for (char digit : token) {
    if (digit < '0' || digit > '9' ||
        n > (uint64_t(kMax) - uint64_t(digit - '0')) / 10)
      return false;
    n = n * 10 + uint64_t(digit - '0');
  }
  *out = int64_t(n);
  return true;
}

} // namespace

Result<P1LiveRoom>
measure_live_room(const Json &facts, const std::filesystem::path &sys_root,
                  const std::vector<std::string> &selected_bdfs) {
  const auto fail = [](std::string path,
                       std::string reason) -> Result<P1LiveRoom> {
    return refuse(std::move(path), std::move(reason));
  };
  if (!facts.is_object() || !facts.contains("host") ||
      !facts.at("host").is_object() || !facts.contains("gpus") ||
      !facts.at("gpus").is_array())
    return fail("facts", "a fresh bc-serve detect() result is required");
  const Json &host = facts.at("host");
  if (!host.contains("mem_available_mib") || !host.contains("memory_cgroups") ||
      !host.contains("amdgpu_pin") || !host.at("memory_cgroups").is_array() ||
      !host.at("amdgpu_pin").is_object())
    return fail(
        "facts",
        "MemAvailable, effective memory cgroups and pin facts are required");

  P1LiveRoom out;
  if (!mib(host.at("mem_available_mib"), &out.mem_available_bytes))
    return fail("mem_available_bytes",
                "live MemAvailable is missing, empty or exceeds I-JSON range");
  out.cgroup_headroom_bytes = out.mem_available_bytes;
  for (const auto &cg : host.at("memory_cgroups")) {
    if (!cg.is_object() || !cg.contains("room_mib"))
      return fail("cgroup_headroom_bytes", "cgroup room is not measured");
    int64_t room = 0;
    if (!mib(cg.at("room_mib"), &room, false))
      return fail("cgroup_headroom_bytes", "invalid measured cgroup room");
    out.cgroup_headroom_bytes = std::min(out.cgroup_headroom_bytes, room);
  }
  if (out.cgroup_headroom_bytes <= 0)
    return fail("cgroup_headroom_bytes", "no live effective cgroup/host room");

  const Json &pin = host.at("amdgpu_pin");
  if (!pin.contains("loaded") || !pin.contains("lifted") ||
      !pin.at("loaded").is_boolean() || !pin.at("lifted").is_boolean())
    return fail("pin_available_bytes",
                "amdgpu pinned-memory cap state was not read");
  if (!pin.at("loaded").get<bool>())
    return fail("pin_available_bytes",
                "amdgpu not loaded; cannot measure pin headroom");

  std::set<std::string> observed;
  for (const Json &gpu : facts.at("gpus")) {
    if (!gpu.is_object() || !gpu.contains("bdf") || !gpu.at("bdf").is_string())
      return fail("inventory", "malformed KFD GPU identity");
    const std::string bdf = gpu.at("bdf").get<std::string>();
    if (!valid_bdf(bdf) || !observed.insert(bdf).second)
      return fail("inventory", "invalid or duplicate selected BDF");
  }
  if (selected_bdfs.empty() ||
      selected_bdfs.size() !=
          std::set<std::string>(selected_bdfs.begin(), selected_bdfs.end())
              .size())
    return fail("inventory",
                "selected engine/helper BDFs missing or duplicated");
  for (const auto &bdf : selected_bdfs)
    if (!valid_bdf(bdf) || !observed.count(bdf))
      return fail("inventory", "selected BDF is absent from fresh KFD facts");

  int64_t effective_pin_cap = 0;
  if (pin.at("lifted").get<bool>()) {
    // No driver cap: live host/cgroup room is still the conservative bound.
    effective_pin_cap = out.cgroup_headroom_bytes;
    out.evidence = "bc-serve detect: amdgpu.no_system_mem_limit is lifted";
  } else {
    if (!pin.contains("cap_mib") || !mib(pin.at("cap_mib"), &effective_pin_cap))
      return fail("pin_available_bytes",
                  "amdgpu cap is missing; no guessed fallback");
    const auto devices = sys_root / "sys/bus/pci/devices";
    std::error_code ec;
    if (!std::filesystem::is_directory(devices, ec))
      return fail("pin_available_bytes", "PCI sysfs device accounting missing");
    int64_t total_gtt = 0;
    size_t seen = 0;
    for (const auto &item : std::filesystem::directory_iterator(devices, ec)) {
      if (ec)
        return fail("pin_available_bytes",
                    "failed to walk PCI device accounting");
      const auto used = item.path() / "mem_info_gtt_used";
      if (!std::filesystem::exists(used, ec)) {
        if (ec)
          return fail("pin_available_bytes", "cannot inspect GTT accounting");
        continue;
      }
      int64_t n = 0;
      if (!bytes_from_file(used, &n) || total_gtt > kMax - n)
        return fail("pin_available_bytes",
                    "GTT usage is missing, invalid or overflows");
      total_gtt += n;
      ++seen;
    }
    if (!seen)
      return fail("pin_available_bytes", "no driver GTT usage counters found");
    for (const auto &bdf : selected_bdfs)
      if (!std::filesystem::exists(devices / bdf / "mem_info_gtt_used"))
        return fail("pin_available_bytes",
                    "selected GPU has no live GTT usage counter");
    if (effective_pin_cap <= total_gtt)
      return fail("pin_available_bytes",
                  "measured GTT usage exhausts global driver pin cap");
    effective_pin_cap -= total_gtt;
    out.evidence = "bc-serve detect: amdgpu pin cap less live PCI GTT usage";
  }
  out.pin_available_bytes =
      std::min(out.cgroup_headroom_bytes, effective_pin_cap);
  if (out.pin_available_bytes <= 0)
    return fail("pin_available_bytes", "no live measured pinnable headroom");
  return out;
}
} // namespace bc::serving::v2
