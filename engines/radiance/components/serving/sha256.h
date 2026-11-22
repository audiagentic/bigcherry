#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace bc::serving {

// Which block function runs. Hardware is the CPU's SHA instructions; Portable is the plain C++ one and is always
// available. Both compute FIPS 180-4 SHA-256 and give the same digest for every input.
enum class Sha256Backend { Portable, Hardware };

// FIPS 180-4 incremental SHA-256. update() retains at most one partial block,
// regardless of the source file's length.
class Sha256 final {
  public:
    // The best backend this CPU has (checked at run time), so one binary runs on hosts without SHA instructions.
    Sha256();
    // A named backend; Hardware throws std::logic_error on a CPU without it.
    explicit Sha256(Sha256Backend backend);
    static bool hardware_available();
    void update(std::string_view bytes);
    std::string hex_digest() const;

  private:
    void process_blocks(const unsigned char* blocks, size_t count);
    std::array<uint32_t, 8> state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::array<unsigned char, 64> pending_{};
    size_t partial_ = 0;
    uint64_t size_ = 0;
    Sha256Backend backend_;
};

// Lowercase SHA-256 of exactly these bytes, equivalent to incremental digest.
std::string sha256_hex(const std::string& text);

} // namespace bc::serving
