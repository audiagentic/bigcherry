#include "v2/p1_live_room.h"

#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace fs = std::filesystem;
using bc::serving::v2::measure_live_room;
using Json = nlohmann::json;
constexpr int64_t M = 1024 * 1024;

int main() {
  const auto suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path root =
      fs::temp_directory_path() / ("bc-p1-live-room-" + std::to_string(suffix));
  const auto devices = root / "sys/bus/pci/devices";
  const std::string engine = "0000:03:00.0";
  const std::string helper = "0000:08:00.0";
  const std::string other = "0000:0a:00.0";
  for (const auto &bdf : {engine, helper, other})
    fs::create_directories(devices / bdf);
  const auto set_gtt = [&](const std::string &bdf, int64_t n) {
    std::ofstream f(devices / bdf / "mem_info_gtt_used");
    f << n << "\n";
    assert(bool(f));
  };
  set_gtt(engine, 2 * M);
  set_gtt(helper, 3 * M);
  set_gtt(other, 5 * M); // global cap counts cards outside this layout

  Json facts = {
      {"host",
       {{"mem_available_mib", 300},
        {"memory_cgroups",
         Json::array({{{"room_mib", 240}, {"dir", "/sys/fs/cgroup"}}})},
        {"amdgpu_pin",
         {{"loaded", true}, {"lifted", false}, {"cap_mib", 220}}}}},
      {"gpus", Json::array({{{"bdf", engine}}, {{"bdf", helper}}})}};
  auto room = measure_live_room(facts, root, {engine, helper});
  assert(room);
  assert(room.value().mem_available_bytes == 300 * M);
  assert(room.value().cgroup_headroom_bytes == 240 * M);
  assert(room.value().pin_available_bytes == 210 * M);
  assert(room.value().evidence.find("GTT") != std::string::npos);

  // Safe live update is sampled afresh; never cached per layout.
  set_gtt(helper, 7 * M);
  room = measure_live_room(facts, root, {engine, helper});
  assert(room);
  assert(room.value().pin_available_bytes == 206 * M);

  facts["host"]["amdgpu_pin"]["lifted"] = true;
  room = measure_live_room(facts, root, {engine, helper});
  assert(room);
  assert(room.value().pin_available_bytes == 240 * M);

  facts["host"]["amdgpu_pin"]["lifted"] = false;
  fs::remove(devices / engine / "mem_info_gtt_used");
  assert(!measure_live_room(facts, root, {engine, helper}));
  set_gtt(engine, 2 * M);
  facts["host"]["amdgpu_pin"]["cap_mib"] = nullptr;
  assert(!measure_live_room(facts, root, {engine, helper}));
  facts["host"]["amdgpu_pin"]["cap_mib"] = 220;
  facts["host"]["memory_cgroups"][0]["room_mib"] = 0;
  assert(!measure_live_room(facts, root, {engine, helper}));
  facts["host"]["memory_cgroups"][0]["room_mib"] = 240;
  assert(!measure_live_room(facts, root, {engine, engine}));
  assert(!measure_live_room(facts, root, {"0000:09:00.0"}));
  facts["host"]["mem_available_mib"] = "300";
  assert(!measure_live_room(facts, root, {engine}));
  fs::remove_all(root);
}
