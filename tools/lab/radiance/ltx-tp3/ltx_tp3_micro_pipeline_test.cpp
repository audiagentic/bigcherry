// LT3 miniature integration harness: one synthetic request through component seams.
// Exercises tags, head ownership, all-head RMS, 3 partials, schedule and leader.
// No native Radiance model, graph, GEMM, HIP collective or media decoding.
#include "ltx_tp3_reference.h"
#include "ltx_tp3_protocol.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace bc_ltx_tp3_reference;
using namespace bc_ltx_tp3_protocol;
namespace {
template<class F> void refuse(F f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
void step(bool remote_text, bool stg, bool iso, Shares shares) {
    const auto gemma = owners(Stage::gemma, remote_text);
    const auto conn = owners(Stage::connectors, remote_text);
    const auto dit = owners(Stage::dit, remote_text);
    const auto decoder = owners(Stage::decoder, remote_text);
    assert(gemma[remote_text ? 3 : 0] && conn[0] && decoder[0]);
    assert(dit[0] && dit[1] && dit[2] && !dit[3]);

    ConditioningMailbox transfer;
    const Ticket ticket{71, 2, 2, 16, 2};
    std::vector<std::uint8_t> payload(payload_bytes(ticket));
    for (std::size_t i = 0; i < payload.size(); ++i)
        payload[i] = std::uint8_t(i * 7);
    transfer.publish(ticket, payload);
    const auto consumed = transfer.receive(ticket);
    assert(consumed == payload);
    refuse([&] { transfer.receive(ticket); });

    const Trace events = collective_trace(stg, iso);
    require_agreement({events, events, events});
    auto mismatched = events;
    mismatched[0] += 1; // injected lost/wrong operation
    refuse([&] { require_agreement({events, mismatched, events}); });

    constexpr int width = 2048;  // audio all-head Q/K, 32 heads x 64
    const auto pieces = attention_slices(32, 64, shares);
    const auto cuts = iq8r_input_slices(width, {
        shares[0] * 64, shares[1] * 64, shares[2] * 64});
    for (int r = 0; r < 3; ++r) {
        assert(pieces[r].offset == cuts[r].offset);
        assert(pieces[r].width == cuts[r].width);
    } // head and IQ8R cuts must agree
    std::array<std::vector<float>, 3> x, gain;
    std::vector<float> dense(width, 0);
    for (int rank = 0; rank < 3; ++rank)
        for (int i = pieces[rank].offset; i < pieces[rank].offset + pieces[rank].width; ++i) {
            const float value = float((i * 19 + 37) % 107 - 53) / 59.0f;
            dense[i] = value;
            x[rank].push_back(value);
            gain[rank].push_back(0.5f + float(i % 11) / 17.0f);
        }
    const auto normalized = global_rms(x, gain, 1e-6);
    double sum_sq = 0, dense_output = 0;
    for (double a : dense) sum_sq += a * a;
    const double scale = 1.0 / std::sqrt(sum_sq / double(width) + 1e-6);
    constexpr double bias = 0.125; // one output-projection bias
    dense_output = bias;
    for (int i = 0; i < width; ++i) {
        const double g = 0.5 + double(i % 11) / 17;
        const double weight = double((i * 7 + 13) % 83 - 41) / 91;
        dense_output += dense[i] * scale * g * weight;
    }
    double sharded_output = bias;
    for (int rank = 0; rank < 3; ++rank)
        for (int j = 0; j < pieces[rank].width; ++j) {
            const int i = pieces[rank].offset + j;
            const double weight = double((i * 7 + 13) % 83 - 41) / 91;
            sharded_output += normalized[rank][j] * weight;
        }
    assert(std::abs(sharded_output - dense_output) < 2e-5);
    // Exactly one owner is allowed to commit the decoded result.
    int media_writes = 0;
    for (int rank = 0; rank < 4; ++rank)
        if (decoder[rank]) ++media_writes;
    assert(media_writes == 1);
}
} // namespace

int main() {
    for (bool remote : {false, true})
        for (bool stg : {false, true})
            for (bool iso : {false, true})
                for (auto shares : {Shares{8, 8, 16}, Shares{12, 12, 8}})
                    step(remote, stg, iso, shares);
    refuse([]{ iq8r_input_slices(2048, {640, 640, 768}); });
    std::cout << "LT3 synthetic component-chain integration: passed\n";
}
