#pragma once
/* bc-serve's v1 manifest writer (RSC08 S2a): the resolver's output, the host facts, the measured plan and the planner's
 * owner map assembled into a serving manifest v1 document (docs/design/serving-manifest-v1), digested and checked.
 *
 * The result is accepted only when the document reads back through the shared strict reader (v1::read_manifest: strict
 * JSON, identity, JCS digest, structure, provenance pointers and coverage, semantics) and its owner map equals the
 * planner's (verify_owner_map). A document that does not is an error carrying the reader's own message, never a file. */
#include "plan_cache.h"
#include "serving.h"

#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

namespace bc::serving {

struct ManifestInputs {
    Resolved resolved;
    std::string profile = "production";      // spec.intent.service.profile
    std::string instance_id;                 // UUID v4
    std::string producer_sha256;             // SHA-256 of the running bc-serve executable
    std::filesystem::path calibration_cache; // the refit layer the resolver used ("" = seeds only)
    // The measured engine plan and what it was measured for (the interim plan cache's entry and its key inputs).
    CachedPlan plan;
    nlohmann::json plan_key_inputs;
};

struct BuiltManifest {
    nlohmann::json document;
    std::string text;    // the document as written: two-space indent, trailing newline
    std::string digest;  // "sha256:" + hex, also in document.metadata.content_digest
};

/* Throws std::runtime_error with the failing step's message. */
BuiltManifest build_manifest(const ManifestInputs& in);

/* Atomically creates `path` read-only with `text`; refuses an existing path (a manifest is immutable and never replaced). */
void write_manifest_file(const std::filesystem::path& path, const std::string& text);

/* A fresh UUID v4 (the kernel's), and the SHA-256 of a file's bytes. */
std::string new_instance_id();
std::string file_sha256(const std::filesystem::path& file);

}  // namespace bc::serving
