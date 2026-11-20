// LT3 component harness: only the authorized leader writes decoded media.
// VAE/VUPSCALE/vocoder kernels are NOT run here.
#include "ltx_tp3_protocol.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <tuple>

using namespace bc_ltx_tp3_protocol;
namespace {
template<class F> void refuse(F f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
enum class Media { first_frame_vae, video_decode, audio_decode, video_upscale };
struct Key {
    std::uint64_t request, epoch;
    Media media;
    auto operator<=>(const Key&) const = default;
};
class MockOutputOwner {
    std::map<Key, int> committed_;
    std::map<std::uint64_t, std::uint64_t> latest_;
    std::map<std::uint64_t, std::uint64_t> maximum_seen_;
public:
    void begin(std::uint64_t request, std::uint64_t epoch) {
        if (!request || !epoch || (maximum_seen_.contains(request) && maximum_seen_[request] >= epoch))
            throw std::invalid_argument("replayed or invalid request");
        latest_[request] = epoch;
        maximum_seen_[request] = epoch;
    }
    void write(Key k, int rank) {
        if (rank != 0 || !latest_.contains(k.request) || latest_[k.request] != k.epoch)
            throw std::invalid_argument("nonleader or stale decode");
        if (!committed_.emplace(k, rank).second)
            throw std::invalid_argument("duplicate media writer");
    }
    void cancel(std::uint64_t request) { latest_.erase(request); }
    int writes(std::uint64_t request) const {
        int count = 0;
        for (const auto& [key, rank] : committed_) {
            assert(rank == 0);
            if (key.request == request) ++count;
        }
        return count;
    }
};
void video_and_audio(bool image_first_frame) {
    MockOutputOwner owner;
    owner.begin(31, 1);
    if (image_first_frame)
        owner.write({31, 1, Media::first_frame_vae}, 0);
    for (auto m : {Media::video_decode, Media::audio_decode, Media::video_upscale}) {
        owner.write({31, 1, m}, 0);
        refuse([&] { owner.write({31, 1, m}, 1); });
        refuse([&] { owner.write({31, 1, m}, 0); });
    }
    assert(owner.writes(31) == (image_first_frame ? 4 : 3));
    owner.cancel(31);
    refuse([&] { owner.write({31, 1, Media::audio_decode}, 0); });
    refuse([&] { owner.begin(31, 1); }); // would replay the same epoch
    owner.begin(31, 2);
    owner.write({31, 2, Media::video_decode}, 0);
    assert(owner.writes(31) == (image_first_frame ? 5 : 4));
}
} // namespace
int main() {
    video_and_audio(false);
    video_and_audio(true);
    std::cout << "LT3 decoder and output ownership: passed\n";
}
