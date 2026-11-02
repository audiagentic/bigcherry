#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace bc::serving {

/* How a host consumer holds its memory, which decides the limits it counts against:
 *   pinned   driver-pinned memory a card reads (a userptr mapping): MemAvailable, every memory cgroup and the
 *            amdgpu pin cap; RLIMIT_MEMLOCK does not count it
 *   mlocked  an mlock()ed copy: MemAvailable, every memory cgroup and RLIMIT_MEMLOCK (unless CAP_IPC_LOCK)
 *   pageable ordinary memory: MemAvailable and every memory cgroup */
enum class Residence { pinned, mlocked, pageable };

struct HostConsumer {
    std::string name;
    uint64_t bytes = 0;
    Residence residence = Residence::pageable;
    std::string from;   // the function that sized it: "engine plan (rad0011: size_checkpoint_store)", ...
    std::string remedy; // what an operator can change to shrink it, named in a refusal; "" for none
};

/* The engine's planning query output (radiance --plan-memory --json, slice 1b), version 1. Strict: unknown keys,
 * missing keys and wrong types are refused. Every consumer names the engine function its bytes came from. */
struct EnginePlan {
    std::string producer, inputs_sha256;
    std::vector<HostConsumer> host;          // host-resident consumers the engine pins or maps at start-up
    uint64_t kv_bytes_per_token = 0;         // one token of every KV group, for the host KV tier's minimum
    std::string kv_from;
};
EnginePlan parse_engine_plan(const nlohmann::json& plan);

struct LedgerInputs {
    nlohmann::json host_facts;               // detect()'s "host" object
    uint64_t reserve_mib = 0;                // memory.host_reserve_mib
    std::vector<HostConsumer> consumers;     // every planned host consumer except the host KV tier
    bool retention_keep = true;              // service.retention: keep finished sessions in the host KV tier
    uint64_t min_tier_bytes = 0;             // below this the tier holds no full session: it is 0
};

struct Ledger {
    uint64_t host_kv_mib = 0;
    nlohmann::json explain;                  // limits, consumers, reserve, tier and why
};

/* The host RAM plan: every consumer and the reserve against each limit it counts under; the host KV tier takes the
 * room left. Refuses (with the largest consumer named) when the consumers and reserve do not fit. */
Ledger host_ledger(const LedgerInputs& in);

}  // namespace bc::serving
