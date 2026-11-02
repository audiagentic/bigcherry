#pragma once
#include <string>

namespace bc::serving {

/* Lowercase hex SHA-256 (FIPS 180-4) of `text`'s bytes. Self-contained, so neither bc_serving nor the contract generator
 * needs a crypto library. */
std::string sha256_hex(const std::string& text);

}  // namespace bc::serving
