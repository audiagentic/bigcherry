#pragma once
/* Strict JSON for v1 documents (spec §8, errors.md; RSC08 slice 2 gates): a size limit, UTF-8 checked before parsing,
 * duplicate members refused while parsing (escape-equivalent names included) with the holding object's path, and
 * every number an integer within I-JSON's +/-(2^53 - 1). Nothing that fails here reaches hashing. */
#include "error.h"

#include <cstddef>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace bc::serving::v1 {

inline constexpr size_t kMaxDocumentBytes = 16u * 1024u * 1024u;
/* spec §11 amendment 7: the v1 examples nest at most 9 deep; 64 bounds the parser's paths and every recursive walk
 * (JCS, the decoder) far from the stack. */
inline constexpr size_t kMaxNestingDepth = 64;

/* The parsed document, or BC_CFG_INVALID_FORMAT (size, encoding, syntax, number) or BC_CFG_DUPLICATE_FIELD, with
 * `source` as the error's source file and, for syntax errors, its line and column. */
Result<nlohmann::json> parse_strict_json(std::string_view bytes, const std::string& source);

}  // namespace bc::serving::v1
