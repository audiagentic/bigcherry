#pragma once
#include <string>
#include <nlohmann/json.hpp>

namespace bc::serving {

/* RFC 8785 JSON Canonicalization Scheme for the serving manifest v2 subset (docs/design/serving-manifest-v2, spec §8):
 * objects with members sorted by their names' UTF-16 code units, no whitespace, strings with JCS's escapes, and numbers
 * that are integers only (v1 has no non-integer number; one is refused, never formatted). Integers must be within
 * I-JSON's ±(2^53 − 1). On success `out` holds the canonical UTF-8 bytes; otherwise `why` names the first offending
 * JSON Pointer and the reason. */
bool jcs_canonicalize(const nlohmann::json& value, std::string* out, std::string* why);

/* "sha256:<64 hex>" over the JCS bytes of `manifest` with metadata.content_digest removed (spec §8). The member may be
 * absent (a producer digests before inserting it); present, it must be a string. Its format is the reader's check.
 * False, with `why`, when the document cannot be canonicalized or the member has another type. */
bool manifest_content_digest(const nlohmann::json& manifest, std::string* digest, std::string* why);

}  // namespace bc::serving
