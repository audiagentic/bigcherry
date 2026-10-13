#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define R3_NGRAM_HELPER_ABI 2u

uint32_t r3_ngram_helper_abi(void);
int r3_ngram_helper_capacity(int device, uint64_t reserve_bytes, uint64_t* usable_bytes);
int r3_ngram_helper_prepare(int device, int64_t first_row, int64_t rows,
                            int n_embd, int max_m);
/* Copies `bytes` of codes to the device at byte `offset` of the shard prepared for it; a shard is loaded in
 * pieces, so the caller never needs the whole of it in host memory at once. */
int r3_ngram_helper_load(int device, int64_t offset, const uint8_t* host_codes, int64_t bytes);
int r3_ngram_helper_gather(int device, const int32_t* ids, int m, int vocab_offset,
                           float scale, uint16_t* out_bf16);
void r3_ngram_helper_release(int device);

#ifdef __cplusplus
}
#endif
