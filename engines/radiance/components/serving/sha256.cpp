#include "sha256.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#if defined(__x86_64__)
#include <cpuid.h>
#include <immintrin.h>
#endif

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


void portable_blocks(std::array<uint32_t, 8>& state_, const unsigned char* block) {
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


#if defined(__x86_64__)
// The CPU's SHA-256 instructions need SHA plus the SSSE3 and SSE4.1 shuffles around them (CPUID leaf 7 EBX bit 29,
// leaf 1 ECX bits 9 and 19). They work on XMM registers only, so no OS state needs enabling.
bool cpu_has_sha() {
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & (1U << 9U)) || !(c & (1U << 19U))) return false;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(7, 0, a, b, c, d);
    return (b & (1U << 29U)) != 0;
}

// Same rounds as portable_blocks, four at a time: the message schedule is sha256msg1/msg2, and the round constants
// are the same kRound table. State is kept in the ABEF/CDGH layout the instructions use.
__attribute__((target("sha,ssse3,sse4.1")))
void hardware_blocks(std::array<uint32_t, 8>& state, const unsigned char* data, size_t count) {
    const __m128i swap = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);
    __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[0]));
    __m128i cdgh = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[4]));
    tmp = _mm_shuffle_epi32(tmp, 0xB1);
    cdgh = _mm_shuffle_epi32(cdgh, 0x1B);
    __m128i abef = _mm_alignr_epi8(tmp, cdgh, 8);
    cdgh = _mm_blend_epi16(cdgh, tmp, 0xF0);
    for (; count; --count, data += 64) {
        const __m128i abef_in = abef, cdgh_in = cdgh;
        __m128i w[4];
#pragma GCC unroll 16
        for (int i = 0; i < 16; ++i) {
            if (i < 4) {
                w[i] = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 16 * i)), swap);
            } else {
                const __m128i& w4 = w[i & 3];       // W[i-4], replaced by W[i]
                const __m128i& w3 = w[(i + 1) & 3];
                const __m128i& w2 = w[(i + 2) & 3];
                const __m128i& w1 = w[(i + 3) & 3];
                w[i & 3] = _mm_sha256msg2_epu32(
                    _mm_add_epi32(_mm_sha256msg1_epu32(w4, w3), _mm_alignr_epi8(w1, w2, 4)), w1);
            }
            __m128i msg = _mm_add_epi32(w[i & 3], _mm_loadu_si128(reinterpret_cast<const __m128i*>(&kRound[4 * i])));
            cdgh = _mm_sha256rnds2_epu32(cdgh, abef, msg);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            abef = _mm_sha256rnds2_epu32(abef, cdgh, msg);
        }
        abef = _mm_add_epi32(abef, abef_in);
        cdgh = _mm_add_epi32(cdgh, cdgh_in);
    }
    tmp = _mm_shuffle_epi32(abef, 0x1B);
    cdgh = _mm_shuffle_epi32(cdgh, 0xB1);
    abef = _mm_blend_epi16(tmp, cdgh, 0xF0);
    cdgh = _mm_alignr_epi8(cdgh, tmp, 8);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[0]), abef);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[4]), cdgh);
}
#else
bool cpu_has_sha() { return false; }
void hardware_blocks(std::array<uint32_t, 8>&, const unsigned char*, size_t) {}
#endif

} // namespace

Sha256::Sha256() : backend_(cpu_has_sha() ? Sha256Backend::Hardware : Sha256Backend::Portable) {}

Sha256::Sha256(Sha256Backend backend) : backend_(backend) {
    if (backend == Sha256Backend::Hardware && !cpu_has_sha())
        throw std::logic_error("hardware SHA-256 is not available on this CPU");
}

bool Sha256::hardware_available() { return cpu_has_sha(); }

void Sha256::process_blocks(const unsigned char* blocks, size_t count) {
    if (backend_ == Sha256Backend::Hardware) {
        hardware_blocks(state_, blocks, count);
        return;
    }
    for (; count; --count, blocks += 64) portable_blocks(state_, blocks);
}

void Sha256::update(std::string_view bytes) {
    if (bytes.size() > (std::numeric_limits<uint64_t>::max() / 8U) - size_)
        throw std::overflow_error("SHA-256 input bit length overflows");
    size_ += static_cast<uint64_t>(bytes.size());
    size_t offset = 0;
    // Whole blocks go straight from the caller's bytes to the block function; only a partial block is buffered.
    if (partial_ == 0) {
        const size_t whole = bytes.size() / pending_.size();
        if (whole) process_blocks(reinterpret_cast<const unsigned char*>(bytes.data()), whole);
        offset = whole * pending_.size();
    }
    while (offset < bytes.size()) {
        const size_t take = std::min<size_t>(pending_.size() - partial_, bytes.size() - offset);
        std::copy_n(reinterpret_cast<const unsigned char*>(bytes.data() + offset), take,
                    pending_.data() + partial_);
        offset += take;
        partial_ += take;
        if (partial_ == pending_.size()) {
            process_blocks(pending_.data(), 1);
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
        final.process_blocks(final.pending_.data(), 1);
        final.partial_ = 0;
    }
    std::fill(final.pending_.begin() + static_cast<std::ptrdiff_t>(final.partial_), final.pending_.begin() + 56,
              0U);
    for (int i = 0; i < 8; ++i)
        final.pending_[56 + i] = static_cast<unsigned char>(bits >> (56U - (8U * static_cast<unsigned>(i))));
    final.process_blocks(final.pending_.data(), 1);
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
