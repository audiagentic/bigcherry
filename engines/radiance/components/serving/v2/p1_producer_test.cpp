#include "v2/p1_producer.h"

#include <cassert>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>

int main() {
  const bc::serving::Resolved empty{};
  const nlohmann::json request = {
      {"engine_plan_file", "/does/not/exist/plan.json"},
      {"rank_demand_file", "/does/not/exist/ranks.json"},
      {"engine_binary", "/does/not/exist/radiance"},
      {"helper_binary", ""},
      {"margin", {{"key", "host_ledger_margin.mib"}, {"source", "cache"},
                  {"evidence", "actual-P2-measurement"}, {"mib", 1}}},
      {"run_id", "native-start"}, {"retention_target_tokens", 128}};
  auto rejected = [&](const nlohmann::json& j, const std::string& reason) {
    try {
      (void)bc::serving::v2::produce_p1_inputs(empty,nlohmann::json::object(),"/",j);
      assert(false && "invalid P1 request must refuse");
    } catch (const std::runtime_error& e) {
      assert(std::string(e.what()).find(reason) != std::string::npos);
    }
  };
  auto bad = request;
  bad.erase("engine_plan_file");
  rejected(bad, "missing or unknown");
  bad = request;
  bad["margin"]["source"] = "guess";
  rejected(bad, "seed or a calibration-cache value");
  bad = request;
  bad["margin"]["evidence"] = "";
  rejected(bad, "margin.evidence");
  rejected(request, "missing measuring-start artifact");   // the request names files that do not exist
}
