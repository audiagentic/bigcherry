#pragma once
// RSC02: optional, launch-bound attestation of a legacy native-endian rank
// histogram. The rank file remains uint32_t[kMaxLayers][expert_count],
// unchanged for workers. Only the owner-injected nonce enables the versioned
// companion proof.
#include "moe_remote_shm.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace r3::heat_publication {

inline constexpr uint64_t kProofMagic = 0x3154465248544342ULL; // "BCHTRFT1" (native endian)
inline constexpr uint32_t kProofVersion = 1;
inline constexpr size_t kNonceLength = 32; // 128-bit owner-generated launch nonce, lowercase hex

struct Proof {
    uint64_t magic = kProofMagic;
    uint32_t version = kProofVersion;
    uint32_t record_size = 88;
    std::array<char, kNonceLength> launch_nonce{};
    uint64_t device = 0;
    uint64_t inode = 0;
    uint64_t size_bytes = 0;
    int64_t modified_sec = 0;
    int64_t modified_nsec = 0;
};
static_assert(sizeof(Proof) == 88 && std::is_trivially_copyable_v<Proof>);

inline bool valid_nonce(std::string_view nonce) {
    if (nonce.size() != kNonceLength)
        return false;
    for (const char c : nonce)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return false;
    return true;
}

inline std::string proof_path(std::string_view rank_path) {
    return std::string(rank_path) + ".bcproof";
}

// This is the engine's actual CPU publication path, also invoked by the
// card-free producer contract tests. No HIP types, compiler extensions or
// hardware operations enter this header; it is shared with HIP_STANDARD 17.
inline bool publish(const std::string& prefix, uint32_t rank, uint32_t experts,
                    const std::vector<uint32_t>& heat, std::string_view launch_nonce = {}) {
    if (!experts || experts > moe_shm::kMaxHeatLdsBytes / sizeof(uint32_t) ||
        heat.size() != size_t(moe_shm::kMaxLayers) * experts ||
        (!launch_nonce.empty() && !valid_nonce(launch_nonce)))
        return false;
    const std::string out = prefix + ".rank" + std::to_string(rank);
    const std::string tmp = out + ".tmp";
    FILE* file = std::fopen(tmp.c_str(), "wb");
    if (!file)
        return false;
    const size_t n = std::fwrite(heat.data(), sizeof(uint32_t), heat.size(), file);
    const bool flushed = std::fflush(file) == 0;
    struct stat recorded{};
    const bool stamped = flushed && ::fstat(::fileno(file), &recorded) == 0 && S_ISREG(recorded.st_mode);
    const bool closed = std::fclose(file) == 0;
    if (n != heat.size() || !stamped || !closed || std::rename(tmp.c_str(), out.c_str()) != 0)
        return false;
    if (launch_nonce.empty())
        return true; // backward-compatible raw histogram for non-owner launches

    Proof proof{};
    std::memcpy(proof.launch_nonce.data(), launch_nonce.data(), kNonceLength);
    proof.device = static_cast<uint64_t>(recorded.st_dev);
    proof.inode = static_cast<uint64_t>(recorded.st_ino);
    proof.size_bytes = static_cast<uint64_t>(recorded.st_size);
    proof.modified_sec = static_cast<int64_t>(recorded.st_mtim.tv_sec);
    proof.modified_nsec = static_cast<int64_t>(recorded.st_mtim.tv_nsec);
    const std::string sidecar = proof_path(out);
    const std::string sidecar_tmp = sidecar + ".tmp";
    FILE* companion = std::fopen(sidecar_tmp.c_str(), "wb");
    if (!companion)
        return false;
    const bool written = std::fwrite(&proof, sizeof(proof), 1, companion) == 1;
    const bool companion_closed = std::fclose(companion) == 0;
    return written && companion_closed && std::rename(sidecar_tmp.c_str(), sidecar.c_str()) == 0;
}

} // namespace r3::heat_publication
