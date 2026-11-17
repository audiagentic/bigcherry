#include "sha256.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace bc::serving {
namespace {

constexpr uint32_t kRound[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

uint32_t rotate(uint32_t value, unsigned bits) {
    return (value >> bits) | (value << (32U - bits));
}

} // namespace

void Sha256::process(const unsigned char* block) {
    uint32_t w[64];
    for (size_t i = 0; i < 16; ++i) {
        const size_t off = i * 4U;
        w[i] = (uint32_t(block[off]) << 24U) | (uint32_t(block[off + 1]) << 16U) |
               (uint32_t(block[off + 2]) << 8U) | uint32_t(block[off + 3]);
    }
    for (size_t i = 16; i < 64; ++i) {
        const uint32_t s0 = rotate(w[i - 15], 7U) ^ rotate(w[i - 15], 18U) ^ (w[i - 15] >> 3U);
        const uint32_t s1 = rotate(w[i - 2], 17U) ^ rotate(w[i - 2], 19U) ^ (w[i - 2] >> 10U);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (size_t i = 0; i < 64; ++i) {
        const uint32_t s1 = rotate(e, 6U) ^ rotate(e, 11U) ^ rotate(e, 25U);
        const uint32_t choice = (e & f) ^ (~e & g);
        const uint32_t t1 = h + s1 + choice + kRound[i] + w[i];
        const uint32_t s0 = rotate(a, 2U) ^ rotate(a, 13U) ^ rotate(a, 22U);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = s0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void Sha256::update(std::string_view bytes) {
    if (bytes.size() > std::numeric_limits<uint64_t>::max() / 8U - size_)
        throw std::overflow_error("SHA-256 input bit length overflows");
    size_ += static_cast<uint64_t>(bytes.size());
    size_t offset = 0;
    while (offset < bytes.size()) {
        const size_t take = std::min<size_t>(pending_.size() - partial_, bytes.size() - offset);
        std::copy_n(reinterpret_cast<const unsigned char*>(bytes.data() + offset), take,
                    pending_.data() + partial_);
        offset += take;
        partial_ += take;
        if (partial_ == pending_.size()) {
            process(pending_.data());
            partial_ = 0;
        }
    }
}

std::string Sha256::hex_digest() const {
    Sha256 final = *this;
    const uint64_t bits = size_ * 8U;
    // Final padding must not be accounted as additional user input. Feed
    // directly into the pending buffer to allow the maximum valid bit length.
    final.pending_[final.partial_++] = 0x80U;
    if (final.partial_ > 56U) {
        std::fill(final.pending_.begin() + static_cast<std::ptrdiff_t>(final.partial_), final.pending_.end(),
                  0U);
        final.process(final.pending_.data());
        final.partial_ = 0;
    }
    std::fill(final.pending_.begin() + static_cast<std::ptrdiff_t>(final.partial_), final.pending_.begin() + 56,
              0U);
    for (int i = 0; i < 8; ++i)
        final.pending_[56 + i] = static_cast<unsigned char>(bits >> (56U - 8U * static_cast<unsigned>(i)));
    final.process(final.pending_.data());
    std::ostringstream out;
    for (uint32_t v : final.state_)
        out << std::hex << std::setfill('0') << std::setw(8) << v;
    return out.str();
}

std::string sha256_hex(const std::string& text) {
    Sha256 digest;
    digest.update(text);
    return digest.hex_digest();
}

} // namespace bc::serving
