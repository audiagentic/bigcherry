// bigcherry: BLAKE2b with personalisation, for dispatch identity hashing.
//
// Standards 5.4 requires 128-bit BLAKE2b digests with per-component `person=`
// prefixes. The prefix is what stops a signature digest and a hardware digest
// over coincidentally identical bytes from colliding into the same key.
//
// This must agree byte-for-byte with Python's
// ``hashlib.blake2b(data, digest_size=16, person=b"...")``, because the tuning
// tools compute keys in Python and the runtime resolves them in C++. A
// divergence would not fail loudly -- it would simply mean every replay lookup
// misses and silently falls back to native. `test-hip-autotune` therefore
// pins several known-answer vectors against the Python implementation.
//
// Self-contained on purpose: llama.cpp bundles no BLAKE2b, and pulling in a
// dependency for ~150 lines of well-specified code is a poor trade.

#pragma once

#include <stddef.h>
#include <stdint.h>

#define GGML_HIP_BLAKE2B_BLOCK_BYTES 128
#define GGML_HIP_BLAKE2B_MAX_DIGEST  64
#define GGML_HIP_BLAKE2B_PERSON_BYTES 16

struct ggml_hip_blake2b_state {
    uint64_t h[8];
    uint64_t t[2];                                  // message byte counter
    uint8_t  buf[GGML_HIP_BLAKE2B_BLOCK_BYTES];
    size_t   buflen;
    size_t   digest_size;
    bool     finalized;
};

// `person` may be NULL. When shorter than 16 bytes it is zero-padded, exactly
// as Python does.
void ggml_hip_blake2b_init(ggml_hip_blake2b_state * state,
                           size_t digest_size,
                           const char * person);

void ggml_hip_blake2b_update(ggml_hip_blake2b_state * state,
                             const void * data, size_t length);

// Writes `digest_size` bytes to `out`.
void ggml_hip_blake2b_final(ggml_hip_blake2b_state * state, void * out);

// One-shot convenience wrapper.
void ggml_hip_blake2b(void * out, size_t digest_size,
                      const void * data, size_t length,
                      const char * person);
