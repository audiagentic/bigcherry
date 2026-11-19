#include "lab_trace_experiment.h"
#include <cassert>
using nlohmann::json;
int main() {
    using bc::serving::lab_lane_trace_only;
    assert(lab_lane_trace_only(json::parse(R"({"diagnostics":{"lanes":{"trace_level":1}}})")));
    assert(lab_lane_trace_only(json::parse(R"({"diagnostics":{"lanes":{"trace_level":37}}})")));
    for (auto s : {
        R"({})", R"({"placement":{"mode":"observe"}})",
        R"({"diagnostics":{"lanes":{"trace_level":0}}})",
        R"({"diagnostics":{"lanes":{"trace_level":-1}}})",
        R"({"diagnostics":{"lanes":{"trace_level":4294967296}}})",
        R"({"diagnostics":{"lanes":{"trace_level":1,"capture_max":1}}})",
        R"({"diagnostics":{"lanes":{"trace_level":true}}})",
        R"({"diagnostics":{"lanes":{"trace_level":"1"}}})",
        R"({"diagnostics":{"lanes":{"trace_level":1},"gemv":{}}})",
        R"({"diagnostics":{"lanes":{"trace_level":1}},"placement":{"mode":"forced"}})"
    }) assert(!lab_lane_trace_only(json::parse(s)));
}
