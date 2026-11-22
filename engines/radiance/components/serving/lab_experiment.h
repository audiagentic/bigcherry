#pragma once
/* The lab placement experiment a quality score runs with (`bc-serve quality ... --lab-experiment FILE`): a forced
 * placement schedule (spec.experiment.placement) and nothing else. The file is the JSON object bc-serve writes into
 * spec.experiment of the lab-profile manifest; it is read once and checked completely before any GPU is touched. */
#include <cstdint>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace bc::serving {

struct LabPlacementExperiment {
    nlohmann::json document;   // the validated spec.experiment object, as the manifest writer takes it
    uint64_t transactions = 0; // placement proposals the schedule issues: steps x repeat
    int64_t helper_limit = 0;  // highest helper index a `move` destination or an `auto_swap` pair names, plus one
};

/* Throws std::runtime_error naming the file and the member when it cannot be read, is not a JSON object holding only
 * `placement`, or its forced schedule fails the manifest's own placement rules (v2::check_placement_experiment). A
 * schedule is `mode: forced_churn` with `forced`; `observe` and the other experiment kinds are not forced placement. */
LabPlacementExperiment read_lab_placement_experiment(const std::filesystem::path& file);

/* The schedule against the plan's helpers: a forced schedule needs at least two, and a move's destination is one of
 * them. Throws std::runtime_error. */
void require_experiment_helpers(const LabPlacementExperiment& experiment, size_t helper_count);

}  // namespace bc::serving
