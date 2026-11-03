#pragma once
/* The shared strict reader of serving manifest v1 (spec §8-§9, errors.md; RSC08 slice 2). One library, linked by
 * bc-serve, radiance and r3_moe_worker: every process reads the same bytes the same way and gets the same typed views.
 *
 * Order (errors.md §2): strict JSON (size, UTF-8, duplicates, integers) -> contract identity (apiVersion, kind,
 * contract_version) -> content digest -> structural decode (generated codec) -> provenance pointers -> semantics.
 * A stale manifest whose hash and semantics are both wrong fails on the digest; a correctly rehashed one with wrong
 * semantics fails with its semantic code. */
#include "error.h"
#include "serving_types.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace bc::serving::v1 {

Result<Manifest> read_manifest(std::string_view bytes, const std::string& source);
/* BC_CFG_IO_FAILURE when the file cannot be read; then read_manifest. */
Result<Manifest> load_manifest(const std::filesystem::path& path);

/* The cross-field rules a decoded manifest must satisfy (spec §2-§6). read_manifest runs it; exposed for producers. */
Status validate_semantics(const Manifest& m);

/* The engine's typed view: the sections it reads, by reference into the manifest it was built from. */
struct EngineView {
    const Intent& intent;
    const Resources& resources;
    const Handoff& handoff;
    const EngineProcess& engine;
    const EngineMemory& memory;
    int64_t kv_backing_floor_ppm;   // the selected kv_backing_floor.fraction_ppm, or 0
};
EngineView engine_view(const Manifest& m);

/* One helper's typed view: its process entry, owned set, the shared handoff, and its effective settings
 * (helper_common, then its role patch, then lab experiment.shape common and role patches; spec §2.2). */
struct HelperView {
    size_t index;
    const HelperProcess& process;
    const OwnedSet& owned;
    const Handoff& handoff;
    HelperCommon settings;
    bool rows_active;   // spec §2.2: the rows kernel runs when the architecture requires it or the policy asks for it
};

/* spec §5.1: the plan cache key, lowercase hex SHA-256 of the JCS key_inputs built from the manifest and the effective
 * helper settings. read_manifest refuses a manifest whose spec.plan.engine_memory.cache_key differs. */
std::string plan_cache_key(const Manifest& m);
/* BC_CFG_UNRESOLVED_REFERENCE when no helper has `role`. */
Result<HelperView> helper_view(const Manifest& m, std::string_view role);

/* One helper's effective settings: helper_common, then its role patch, then the lab experiment shape's common and role
 * patches (spec §2.2). Needs no owner map, so a producer can call it before it has one. */
HelperCommon effective_helper_settings(const Manifest& m, size_t helper);

/* spec §7 and §11 amendment 6: the document-identity leaves exempt from a provenance record of their own (the root ""
 * record covers them), by name. */
inline constexpr std::array<std::string_view, 7> kIdentityExempt = {
    "/apiVersion", "/kind", "/metadata/instance_id", "/metadata/contract_version", "/metadata/producer/name",
    "/metadata/producer/build_sha256", "/metadata/content_digest"};

/* Identity-envelope coverage (amendment 6): every leaf of the document's /apiVersion, /kind and /metadata that is not
 * exempt has a provenance record at the leaf or at a proper ancestor other than "". Fails closed: a /metadata leaf a
 * future version adds needs a record. Run by read_manifest on the raw document. */
Status check_identity_coverage(const nlohmann::json& doc);

/* spec §7 coverage of the effective configuration (amendment 9): every leaf under /spec has a provenance record at the
 * leaf or a proper ancestor, or else is covered by the root "" record alone, which is only a claim of the schema default:
 * the leaf must then have a registry default (a field's `default`, verified against its source at generation time) and
 * equal it. A record at a non-root pointer must be of the class its covered leaves have in the registry (it covers only
 * homogeneous leaves), and a record whose origin is schema_default must be true: a default exists and the leaf equals it.
 * BC_CFG_PROVENANCE_INVALID naming the leaf otherwise. Run by read_manifest on the raw document. */
Status check_provenance_coverage(const nlohmann::json& doc);

/* spec §4: the owner-map digest segment of helper i. */
std::string owner_segment(size_t helper, const std::vector<std::array<int64_t, 2>>& experts);

}  // namespace bc::serving::v1
