#pragma once
// Included only by r3_moe_worker.cpp. The helper's process-wide settings: a local snapshot of this helper's typed manifest
// section (worker_settings.h), validated by the shared v1 reader before any GPU work. There is no environment parsing here;
// the RHP policy, forced-move and epoch-cycle controls v1 does not model are read in worker_policy_config.h.
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
    uint32_t epoch_cycle_s = 0;
    bool force_enabled = false;
    r3::policy::Settings policy;
    r3::forced::Schedule force_schedule;

    explicit WorkerConfig(const bc::serving::HelperSettings& s)
        : helper_count(s.helper.count), helper_index(s.helper.index), engine_ranks(s.helper.engine_ranks),
          cpu(s.helper.cpu), power_save(s.helper.power_save), queue_warm(s.helper.queue_warm),
          combine(s.helper.combine), combine_min_rows(s.helper.combine_min_rows),
          combine_threads(s.helper.combine_threads), cpu_set_auto(s.helper.cpu_set_auto) {}

    /* The controls v1 does not model (worker_policy_config.h); false with *err naming the variable. */
    bool read_unmodelled(const std::vector<uint32_t>& shares, std::string* err) {
        r3::policy::UnmodelledEnvironment u;
        if (!u.read(helper_index, shares, err)) return false;
        epoch_cycle_s = u.epoch_cycle_s;
        force_enabled = u.force_enabled;
        policy = std::move(u.policy);
        force_schedule = std::move(u.force_schedule);
        return true;
    }
};
