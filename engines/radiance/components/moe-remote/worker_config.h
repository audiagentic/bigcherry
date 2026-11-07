#pragma once
// Included only by r3_moe_worker.cpp. The helper's process-wide settings: a local snapshot of this helper's typed manifest
// section (worker_settings.h), validated by the shared v1 reader before any GPU work. The helper reads no environment
// setting of its own (the one environment write is GPU_MAX_HW_QUEUES, before HIP initialises).
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include "worker_forced_move.h"
#include "worker_policy_config.h"
#include "worker_settings.h"

struct WorkerConfig {
    uint32_t helper_count, helper_index, engine_ranks;
    bool cpu, power_save, queue_warm, combine;
    uint32_t combine_min_rows, combine_threads;
    bool cpu_set_auto;
    /* The no-op placement epoch cycle (RHP11 validation): helper 0 closes, drains, recommits the same placement and reopens
     * every N seconds of burst; 0 = off. */
    uint32_t epoch_cycle_s = 0;
    bool force_enabled = false;
    r3::policy::Settings policy;
    r3::forced::Schedule force_schedule;

    explicit WorkerConfig(const bc::serving::HelperSettings& s)
        : helper_count(s.helper.count), helper_index(s.helper.index), engine_ranks(s.helper.engine_ranks),
          cpu(s.helper.cpu), power_save(s.helper.power_save), queue_warm(s.helper.queue_warm),
          combine(s.helper.combine), combine_min_rows(s.helper.combine_min_rows),
          combine_threads(s.helper.combine_threads), cpu_set_auto(s.helper.cpu_set_auto) {}

    /* The live placement policy, the forced schedule and the epoch cycle of the manifest. False with *err naming the member
     * when an invariant the policy relies on does not hold. */
    bool configure_placement(const bc::serving::HelperSettings& s, std::string* err) {
        if (s.live_policy) {
            std::vector<uint32_t> shares(s.shares.begin(), s.shares.end());
            if (!policy.from_manifest(*s.live_policy, shares, err)) return false;
        }
        if (s.forced) {
            force_schedule.repeat = uint32_t(s.forced->repeat);
            if (s.forced->pace_requests) force_schedule.pace_requests = uint32_t(*s.forced->pace_requests);
            for (const bc::serving::v1::ForcedStep& st : s.forced->steps) force_schedule.steps.push_back(step_of(st));
            force_enabled = true;
        }
        epoch_cycle_s = s.epoch_cycle_s;
        return true;
    }

private:
    static r3::forced::Step step_of(const bc::serving::v1::ForcedStep& st) {
        using bc::serving::v1::ForcedKind;
        const uint32_t layer = uint32_t(st.a_layer), expert = uint32_t(st.a_expert);
        switch (st.kind) {
            case ForcedKind::kMove: return r3::forced::Step(r3::forced::Move{layer, expert, uint32_t(*st.destination)});
            case ForcedKind::kSwap: return r3::forced::Step::swap(layer, expert, uint32_t(*st.b_layer), uint32_t(*st.b_expert));
            case ForcedKind::kAutoSwap: return r3::forced::Step::automatic(layer, expert);
        }
        return r3::forced::Step{};
    }
};
