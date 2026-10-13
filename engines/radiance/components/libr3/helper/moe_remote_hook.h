#pragma once

#include <cstdint>

struct R3MoeMailboxDevice {
    void* mailbox = nullptr;
    const uint8_t* owner = nullptr;
    uint32_t* sequence = nullptr;
    uint32_t rank = 0;
};

extern "C" int r3_moe_mailbox_enabled(void);
extern "C" int r3_moe_mailbox_device(int rank, int world_size, R3MoeMailboxDevice** out);
extern "C" int r3_moe_mailbox_schema_count(void);
