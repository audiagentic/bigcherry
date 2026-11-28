#pragma once
// Native YAML projection shared by admission, resolver, and manifest reconciliation.
// Values are parsed by Radiance, not by a second configuration parser.
namespace bc::serving {
struct NativeTuningField { const char* name; bool integer; };
inline constexpr NativeTuningField kNativeTuningFields[] = {
    {"p2p", false}, {"pipelined_prefill", false}, {"pipelined_prefill_min_rows", true},
    {"prefill_governor", false}, {"tp_ownership", false}, {"tune_pin", false},
    {"attn_prefill_fp8", false}, {"busy_prefill_chunk", false},
    {"state_cache_dtype", false}, {"tp_wire_min_kb", true},
};
}
