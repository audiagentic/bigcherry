// bigcherry: BLAKE2b implementation (RFC 7693) with personalisation.
//
// Compiled unconditionally -- it is plain host C++ with no HIP dependency, and
// the test suite needs it even in builds without the dispatch layer.

#include "hip-autotune-blake2b.h"

#include <string.h>

static const uint64_t GGML_HIP_BLAKE2B_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
};

static const uint8_t GGML_HIP_BLAKE2B_SIGMA[12][16] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
};

static inline uint64_t ggml_hip_rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

static inline uint64_t ggml_hip_load64(const uint8_t * src) {
    // Explicit little-endian assembly rather than a memcpy: the digest must be
    // identical on any host, and this is not hot code.
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= (uint64_t) src[i] << (8 * i);
    }
    return value;
}

static inline void ggml_hip_store64(uint8_t * dst, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        dst[i] = (uint8_t) (value >> (8 * i));
    }
}

#define GGML_HIP_B2B_G(r, i, a, b, c, d)                       \
    do {                                                       \
        a = a + b + m[GGML_HIP_BLAKE2B_SIGMA[r][2 * (i) + 0]]; \
        d = ggml_hip_rotr64(d ^ a, 32);                        \
        c = c + d;                                             \
        b = ggml_hip_rotr64(b ^ c, 24);                        \
        a = a + b + m[GGML_HIP_BLAKE2B_SIGMA[r][2 * (i) + 1]]; \
        d = ggml_hip_rotr64(d ^ a, 16);                        \
        c = c + d;                                             \
        b = ggml_hip_rotr64(b ^ c, 63);                        \
    } while (0)

static void ggml_hip_blake2b_compress(ggml_hip_blake2b_state * state,
                                      const uint8_t block[GGML_HIP_BLAKE2B_BLOCK_BYTES],
                                      bool last) {
    uint64_t m[16];
    uint64_t v[16];

    for (int i = 0; i < 16; ++i) {
        m[i] = ggml_hip_load64(block + i * 8);
    }
    for (int i = 0; i < 8; ++i) {
        v[i]     = state->h[i];
        v[i + 8] = GGML_HIP_BLAKE2B_IV[i];
    }

    v[12] ^= state->t[0];
    v[13] ^= state->t[1];
    if (last) {
        v[14] = ~v[14];
    }

    for (int r = 0; r < 12; ++r) {
        GGML_HIP_B2B_G(r, 0, v[0], v[4], v[ 8], v[12]);
        GGML_HIP_B2B_G(r, 1, v[1], v[5], v[ 9], v[13]);
        GGML_HIP_B2B_G(r, 2, v[2], v[6], v[10], v[14]);
        GGML_HIP_B2B_G(r, 3, v[3], v[7], v[11], v[15]);
        GGML_HIP_B2B_G(r, 4, v[0], v[5], v[10], v[15]);
        GGML_HIP_B2B_G(r, 5, v[1], v[6], v[11], v[12]);
        GGML_HIP_B2B_G(r, 6, v[2], v[7], v[ 8], v[13]);
        GGML_HIP_B2B_G(r, 7, v[3], v[4], v[ 9], v[14]);
    }

    for (int i = 0; i < 8; ++i) {
        state->h[i] ^= v[i] ^ v[i + 8];
    }
}

void ggml_hip_blake2b_init(ggml_hip_blake2b_state * state,
                           size_t digest_size,
                           const char * person) {
    if (digest_size == 0 || digest_size > GGML_HIP_BLAKE2B_MAX_DIGEST) {
        digest_size = GGML_HIP_BLAKE2B_MAX_DIGEST;
    }

    memset(state, 0, sizeof(*state));
    state->digest_size = digest_size;

    // RFC 7693 section 2.8 parameter block, laid out byte-wise and XORed into
    // the IV. Only the fields Python varies are non-default here: digest
    // length, fanout, depth, and the 16-byte personalisation at offset 48.
    uint8_t param[64];
    memset(param, 0, sizeof(param));
    param[0] = (uint8_t) digest_size; // digest_length
    param[1] = 0;                     // key_length -- unkeyed
    param[2] = 1;                     // fanout
    param[3] = 1;                     // depth

    if (person != nullptr) {
        for (size_t i = 0; i < GGML_HIP_BLAKE2B_PERSON_BYTES; ++i) {
            // Stop at the terminator and leave the rest zero: this is exactly
            // Python's zero-padding of a short `person=`.
            if (person[i] == '\0') {
                break;
            }
            param[48 + i] = (uint8_t) person[i];
        }
    }

    for (int i = 0; i < 8; ++i) {
        state->h[i] = GGML_HIP_BLAKE2B_IV[i] ^ ggml_hip_load64(param + i * 8);
    }
}

static inline void ggml_hip_blake2b_increment(ggml_hip_blake2b_state * state,
                                              uint64_t amount) {
    state->t[0] += amount;
    if (state->t[0] < amount) {
        state->t[1] += 1;
    }
}

void ggml_hip_blake2b_update(ggml_hip_blake2b_state * state,
                             const void * data, size_t length) {
    const uint8_t * in = (const uint8_t *) data;

    while (length > 0) {
        // A full buffer is only compressed once we know more input follows:
        // the final block must be compressed with the `last` flag, so it can
        // never be flushed eagerly here.
        if (state->buflen == GGML_HIP_BLAKE2B_BLOCK_BYTES) {
            ggml_hip_blake2b_increment(state, GGML_HIP_BLAKE2B_BLOCK_BYTES);
            ggml_hip_blake2b_compress(state, state->buf, false);
            state->buflen = 0;
        }

        const size_t space = GGML_HIP_BLAKE2B_BLOCK_BYTES - state->buflen;
        const size_t take  = length < space ? length : space;
        memcpy(state->buf + state->buflen, in, take);
        state->buflen += take;
        in            += take;
        length        -= take;
    }
}

void ggml_hip_blake2b_final(ggml_hip_blake2b_state * state, void * out) {
    if (state->finalized) {
        return;
    }
    state->finalized = true;

    ggml_hip_blake2b_increment(state, (uint64_t) state->buflen);
    memset(state->buf + state->buflen, 0,
           GGML_HIP_BLAKE2B_BLOCK_BYTES - state->buflen);
    ggml_hip_blake2b_compress(state, state->buf, true);

    uint8_t full[GGML_HIP_BLAKE2B_MAX_DIGEST];
    for (int i = 0; i < 8; ++i) {
        ggml_hip_store64(full + i * 8, state->h[i]);
    }
    memcpy(out, full, state->digest_size);
}

void ggml_hip_blake2b(void * out, size_t digest_size,
                      const void * data, size_t length,
                      const char * person) {
    ggml_hip_blake2b_state state;
    ggml_hip_blake2b_init(&state, digest_size, person);
    ggml_hip_blake2b_update(&state, data, length);
    ggml_hip_blake2b_final(&state, out);
}
