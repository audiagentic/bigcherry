#pragma once
// RSC11 S13 (inactive P1): sample live host/cgroup and AMD pinned-room facts.
// This is independent of v1 manifest policy and produces no startup settings.
#include "v2/error.h"
#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace bc::serving::v2 {

struct P1LiveRoom {
  int64_t mem_available_bytes = 0;
  int64_t cgroup_headroom_bytes = 0;
  int64_t pin_available_bytes = 0;
  std::string evidence;
};

// 'facts' must come from a fresh bc::serving::detect() for this same sysroot.
// Selected BDFs are used for membership checks; every AMD adapter is accounted
// for in the system-wide GTT consumption (no per-layout undercount).
Result<P1LiveRoom>
measure_live_room(const nlohmann::json &facts,
                  const std::filesystem::path &sys_root,
                  const std::vector<std::string> &selected_bdfs);
} // namespace bc::serving::v2
