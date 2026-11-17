#pragma once
// Immutable, typed engine settings populated once by the strict serving manifest
// reader. DSO consumers look up the exported symbol; no R3_* environment reads.
#include "callback_boundary_abi.h"
#include <cstdint>

struct BcEngineSettings {
    uint32_t struct_size = sizeof(BcEngineSettings);
    uint32_t manifest = 0;
    uint32_t handoff = 0;
    uint32_t skip_owned = 0;
    uint32_t shadow = 0;
    uint32_t mask_only = 0;
    uint32_t combine = 0;
    int64_t max_rows = 64;
    const char* shm = "/r3-moe-helper";
    const char* histogram_file = nullptr;
    // Ephemeral launch proof, decoded once from --bc-heat-launch-nonce, not a manifest field.
    const char* heat_launch_nonce = nullptr;
    int64_t histogram_period_s = 0;
    uint32_t moe_native = 1;
    uint32_t moe_fp16 = 1;
    uint32_t moe_ahalf = 1;
    uint32_t gemv = 0;
    uint32_t dec11 = 0;
    uint32_t fuse_stage = 1;
    uint32_t attn_native = 1;
    uint32_t all_reduce_wire4 = 0;
};

// The engine exports this symbol. Missing symbol or wrong ABI is inert for a
// standalone kernel selftest: the remote handoff remains off, not environment-controlled.
extern "C" int bc_serving_engine_settings(BcEngineSettings* out);

inline BcEngineSettings bc_engine_settings_snapshot() {
    BcEngineSettings settings{};
    static const auto get =
        bc::serving::callback::GuardedSymbol<int(BcEngineSettings*)>::lookup("bc_serving_engine_settings");
    const auto accepted = get.invoke(&settings);
    if (accepted && *accepted != 0 && settings.struct_size == sizeof(settings) && settings.manifest)
        return settings;
    return BcEngineSettings{};
}
