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

/* The engine's planning query output (radiance --plan-memory FILE, managed patch rad0011), version 1: what
 * Engine::plan() sized. Strict: unknown keys, missing keys and wrong types are refused. Every part names the engine
 * function its bytes came from. */
struct EnginePlan {
    std::string producer;
    std::vector<HostConsumer> host;          // host-resident consumers the engine pins at start-up
    uint64_t checkpoint_slot_bytes = 0, checkpoint_floor_slots = 0, checkpoint_want_slots = 0;
    std::string checkpoints_from;
    uint64_t kv_one_session_bytes = 0;       // all fixed state plus pages for one full-length context
    std::string kv_one_session_from;
    nlohmann::json vram;                     // the VRAM split (vram_budget_resolve) and per-card facts, validated
    uint64_t host_pool_mib = 0;              // the host pool the engine was configured with
};
EnginePlan parse_engine_plan(const nlohmann::json& plan);

/* The engine's checkpoint store: pinned linear-state snapshots it sizes itself at start-up from the room the plan
 * leaves (size_checkpoint_store). The ledger plans its size before the host KV tier takes the rest. */
struct CheckpointStore {
    uint64_t slot_bytes = 0, floor_slots = 0, want_slots = 0;
    std::string from;
    // calibration.checkpoint_share: "want" plans want_slots; "floor" plans floor_slots; "tier_first" gives the host KV
    // tier its calibrated target first and the store the remainder, between floor and want.
    std::string share = "want";
    std::string share_from;
};

struct LedgerInputs {
    nlohmann::json host_facts;               // detect()'s "host" object
    uint64_t reserve_mib = 0;                // memory.host_reserve_mib
    std::vector<HostConsumer> consumers;     // every planned host consumer except checkpoints and the host KV tier
    CheckpointStore checkpoints;
    bool retention_keep = true;              // service.retention: keep finished sessions in the host KV tier
    uint64_t min_tier_bytes = 0;             // below this the tier holds no full session: it is 0
    uint64_t tier_target_bytes = 0;          // calibration.host_kv_tier.target_mib, used by share tier_first
};

struct Ledger {
    uint64_t host_kv_mib = 0;
    uint64_t checkpoint_slots = 0;
    nlohmann::json explain;                  // limits, consumers, checkpoints, reserve, tier and why
};

/* The host RAM plan: every consumer, the checkpoint store and the reserve against each limit it counts under; the
 * host KV tier takes the room left. The store is planned at its calibrated share (want by default); when that would
 * leave the tier less than one full-length session it shrinks toward its floor first, and only then does the tier
 * fall to 0. Refuses (with the largest consumer named) when the consumers, the floor and the reserve do not fit. */
Ledger host_ledger(const LedgerInputs& in);

}  // namespace bc::serving
