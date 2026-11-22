// The runtime identity hash has a portable SHA-256 and, where the CPU has them, hardware SHA instructions. Both must
// give the digest FIPS 180-4 defines, for every length and every way of cutting the input into update() calls.
#include "sha256.h"

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

using bc::serving::Sha256;
using bc::serving::Sha256Backend;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

std::string digest(Sha256Backend backend, std::string_view bytes, size_t cut) {
    Sha256 d(backend);
    if (cut == 0) cut = bytes.size() ? bytes.size() : 1;
    for (size_t at = 0; at < bytes.size(); at += cut) d.update(bytes.substr(at, cut));
    return d.hex_digest();
}

// Deterministic pseudo-random bytes (xorshift), so a failure reproduces.
std::string random_bytes(size_t n, uint64_t seed) {
    std::string out(n, '\0');
    uint64_t x = seed | 1U;
    for (char& c : out) {
        x ^= x << 13U;
        x ^= x >> 7U;
        x ^= x << 17U;
        c = static_cast<char>(x >> 24U);
    }
    return out;
}

} // namespace

int main() {
    // FIPS 180-4 / NIST example vectors pin the portable path.
    check(digest(Sha256Backend::Portable, "", 0) ==
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty");
    check(digest(Sha256Backend::Portable, "abc", 0) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc");
    check(digest(Sha256Backend::Portable, "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 0) ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "two-block message");
    check(digest(Sha256Backend::Portable, std::string(1000000, 'a'), 0) ==
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "one million a");

    check(Sha256().hex_digest() == digest(Sha256::hardware_available() ? Sha256Backend::Hardware
                                                                       : Sha256Backend::Portable, "", 0),
          "default backend is the best available one");
    if (!Sha256::hardware_available()) {
        std::printf("hardware SHA not available on this CPU: portable path only\n");
        return failures ? 1 : 0;
    }
    // Hardware must agree with portable for every length across several block boundaries and every cut.
    const std::string data = random_bytes(4 * 64 + 70, 0x9e3779b97f4a7c15ULL);
    for (size_t n = 0; n <= data.size(); ++n) {
        const std::string_view in(data.data(), n);
        const std::string want = digest(Sha256Backend::Portable, in, 0);
        for (size_t cut : {size_t(0), size_t(1), size_t(7), size_t(63), size_t(64), size_t(65), size_t(129)})
            check(digest(Sha256Backend::Hardware, in, cut) == want,
                  "length " + std::to_string(n) + " cut " + std::to_string(cut));
    }
    // A large input in the 64 KiB chunks the runtime identity reads, plus lengths either side of a chunk.
    const size_t chunk = 64 * 1024;
    for (size_t n : {chunk - 1, chunk, chunk + 1, 3 * chunk + 17}) {
        const std::string big = random_bytes(n, n);
        check(digest(Sha256Backend::Hardware, big, chunk) == digest(Sha256Backend::Portable, big, chunk),
              "chunked " + std::to_string(n));
    }
    check(digest(Sha256Backend::Hardware, std::string(1000000, 'a'), 0) ==
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "hardware one million a");
    if (!failures) std::printf("sha256: portable and hardware agree\n");
    return failures ? 1 : 0;
}
