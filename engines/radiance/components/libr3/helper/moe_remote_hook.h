#pragma once

#include <cstdint>

struct R3MoeShadowLayerStats {
    uint64_t rows = 0;
    uint64_t elems = 0;
    float sum_abs = 0.0f;
    float diff2 = 0.0f;
    float ref2 = 0.0f;
    uint32_t max_abs_bits = 0;
    uint32_t bad_rows = 0;
};

struct R3MoeShadowFirst {
    uint32_t claimed = 0;
    uint32_t layer = 0;
    uint32_t rank = 0;
    uint32_t row_base = 0;
    int32_t token = -1;
    uint16_t slot = 0;
    uint16_t reserved0 = 0;
    int32_t expert = -1;
    float rel_l2 = 0.0f;
    float max_abs = 0.0f;
    float local[4]{};
    float helper[4]{};
};

struct R3MoeMailboxDevice {
    void* mailbox = nullptr;
    const uint8_t* owner = nullptr;
    uint32_t* sequence = nullptr;   /* [2]: per-lane-class request counters (small, large) */
    uint32_t* published = nullptr;
    uint16_t* mask_ref = nullptr;
    int32_t* inv = nullptr;     /* [max_rows * top_k] engine sorted row of each locally computed slot, else -1. */
    uint32_t* rmask = nullptr;  /* [max_rows * top_k] 1 where the helper computed the slot. */
    R3MoeShadowLayerStats* shadow_stats = nullptr;
    R3MoeShadowFirst* shadow_first = nullptr;
    uint32_t max_rows = 0;
    uint32_t rank = 0;
    uint32_t world = 0;   /* engine TP ranks: 1 asks the helper for full expert width, 2 for this rank's slice */
    uint32_t shadow = 0;
    uint32_t mask_only = 0;
    uint32_t cpu_ref = 0;
};

extern "C" int r3_moe_mailbox_enabled(void);
extern "C" int r3_moe_mailbox_device(int rank, int world_size, R3MoeMailboxDevice** out);
extern "C" int r3_moe_mailbox_schema_count(void);
