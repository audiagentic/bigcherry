#pragma once
// The legacy WorkerConfig field names are a local snapshot of the manifest's
// typed per-process helper configuration; no environment parsing exists.
#include "worker_settings.h"
struct WorkerConfig {
    uint32_t helper_count, helper_index, engine_ranks;
    bool cpu, power_save, queue_warm, combine;
    uint32_t combine_min_rows, combine_threads;
    bool cpu_set_auto;
    explicit WorkerConfig(const bc::serving::Helper& h)
        : helper_count(h.count), helper_index(h.index), engine_ranks(h.engine_ranks),
          cpu(h.cpu), power_save(h.power_save), queue_warm(h.queue_warm),
          combine(h.combine), combine_min_rows(h.combine_min_rows),
          combine_threads(h.combine_threads), cpu_set_auto(h.cpu_set_auto) {}
};
