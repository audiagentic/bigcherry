#include "lab_experiment.h"

#include "serving_codec.h"
#include "v2/manifest.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace bc::serving {

LabPlacementExperiment read_lab_placement_experiment(const std::filesystem::path& file) {
    const std::string name = file.string();
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("--lab-experiment: cannot read " + name);
    std::ostringstream text;
    text << in.rdbuf();
    LabPlacementExperiment out;
    out.document = nlohmann::json::parse(text.str(), nullptr, false);
    if (!out.document.is_object())
        throw std::runtime_error("--lab-experiment: " + name + " is not a JSON object");
    if (out.document.size() != 1 || !out.document.contains("placement"))
        throw std::runtime_error("--lab-experiment: " + name +
                                 " must hold exactly `placement` (a forced placement schedule); other experiment "
                                 "kinds change the measured plan");
    v2::ExperimentPlacement placement;
    if (const auto e = v2::decode(out.document.at("placement"), "placement", &placement))
        throw std::runtime_error("--lab-experiment: " + name + ": " + e->path + ": " + e->message);
    if (placement.mode != v2::PlacementMode::kForcedChurn || !placement.forced || placement.epoch_cycle_s)
        throw std::runtime_error("--lab-experiment: " + name +
                                 ": placement must be mode forced_churn with `forced` steps and no epoch_cycle_s");
    if (const auto e = v2::check_placement_experiment(placement, size_t{INT64_MAX}))
        throw std::runtime_error("--lab-experiment: " + name + ": " + e->path + ": " + e->message);
    out.transactions = uint64_t(placement.forced->steps.size()) * uint64_t(placement.forced->repeat);
    for (const auto& step : placement.forced->steps) {
        if (step.kind == v2::ForcedKind::kMove)
            out.helper_limit = std::max(out.helper_limit, *step.destination + 1);
        if (step.kind == v2::ForcedKind::kAutoSwap) {
            // an auto_swap carries the two helpers it exchanges between (x, y) in a_layer, a_expert
            if (step.a_layer == step.a_expert)
                throw std::runtime_error("--lab-experiment: " + name + ": an auto_swap names two different helpers");
            out.helper_limit = std::max({out.helper_limit, step.a_layer + 1, step.a_expert + 1});
        }
    }
    return out;
}

void require_experiment_helpers(const LabPlacementExperiment& experiment, size_t helper_count) {
    if (helper_count < 2)
        throw std::runtime_error("--lab-experiment: a forced placement schedule needs a layout with at least two helpers");
    if (experiment.helper_limit > static_cast<int64_t>(helper_count))
        throw std::runtime_error("--lab-experiment: the schedule names helper " +
                                 std::to_string(experiment.helper_limit - 1) + ", but the layout has " +
                                 std::to_string(helper_count) + " helper(s)");
}

}  // namespace bc::serving
