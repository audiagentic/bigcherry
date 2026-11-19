// LT3 laboratory protocol model only: no HIP, engine runtime, GPU IPC or actual wire format.
// Numerically/tracing representative of invariants, not a distributed implementation.
#ifndef BC_LTX_TP3_PROTOCOL_H
#define BC_LTX_TP3_PROTOCOL_H
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bc_ltx_tp3_protocol {

constexpr int kDitBlocks = 48;
constexpr int kStgBlock = 28;
enum class Stage { gemma, connectors, dit, decoder };
enum class Operation { q_norm = 1, k_norm = 2, attention_out = 3, ffn_down = 4 };

// Rank IDs 0..2 are the DiT triplet; rank 3 is never admitted to the DiT.
inline std::array<bool, 4> owners(Stage s, bool text_6900) {
    switch (s) {
        case Stage::gemma: return text_6900 ? std::array<bool, 4>{false,false,false,true}
                                            : std::array<bool, 4>{true,false,false,false};
        case Stage::connectors: return {true,false,false,false};
        case Stage::dit: return {true,true,true,false};
        case Stage::decoder: return {true,false,false,false};
    }
    throw std::invalid_argument("invalid stage");
}

// Required synchronization events for one LTX DiT guidance evaluation.
// Encodes distinct identifiers for each (layer, attention kind, op), so a
// rank that skips one norm or one attention cannot accidentally match.
// This is a schedule oracle, NOT a replacement for the real engine op graph.
using Trace = std::vector<int>;
inline Trace collective_trace(bool stg, bool isolated, int blocks=kDitBlocks) {
    if (blocks != kDitBlocks) throw std::invalid_argument("unexpected LTX block count");
    Trace events;
    auto emit = [&](int block, int kind, Operation op) {
        events.push_back(block * 100 + kind * 10 + static_cast<int>(op));
    };
    for (int l=0; l<blocks; ++l) {
        // Self-attention: video kind 0, audio kind 1. STG bypasses Q/K
        // for both streams at block 28, but still needs output combination.
        for (int stream=0; stream<2; ++stream) {
            if (!(stg && l == kStgBlock)) {
                emit(l, stream, Operation::q_norm);
                emit(l, stream, Operation::k_norm);
            }
            emit(l, stream, Operation::attention_out);
        }
        // Text cross-attention: always present in STG and isolated branches.
        for (int stream=2; stream<4; ++stream) {
            emit(l, stream, Operation::q_norm);
            emit(l, stream, Operation::k_norm);
            emit(l, stream, Operation::attention_out);
        }
        // Cross-modal: isolated guidance omits both directions.
        if (!isolated) {
            for (int stream=4; stream<6; ++stream) {
                emit(l, stream, Operation::q_norm);
                emit(l, stream, Operation::k_norm);
                emit(l, stream, Operation::attention_out);
            }
        }
        emit(l, 6, Operation::ffn_down);
        emit(l, 7, Operation::ffn_down);
    }
    return events;
}
inline void require_agreement(const std::array<Trace, 3>& ranks) {
    if (ranks[0].empty() || ranks[0]!=ranks[1] || ranks[1]!=ranks[2])
        throw std::invalid_argument("TP3 rank collective schedule disagreement");
}

inline std::vector<float> reduce_partials(
        const std::array<std::vector<float>, 3>& partial,
        const std::vector<float>& bias) {
    const auto size=bias.size();
    if (!size) throw std::invalid_argument("empty projection");
    std::vector<float> output(size);
    for (std::size_t i=0; i<size; ++i) {
        output[i]=bias[i];
        for (const auto& rank : partial) {
            if (rank.size()!=size) throw std::invalid_argument("bad rank projection shape");
            output[i]+=rank[i];
        }
    }
    return output;
}

struct Ticket {
    std::uint64_t request = 0;
    std::uint64_t epoch = 0;
    int rows = 0;
    int columns = 0;
    int element_bytes = 0;
};
inline bool same_ticket(const Ticket& a, const Ticket& b) {
    return a.request==b.request && a.epoch==b.epoch &&
           a.rows==b.rows && a.columns==b.columns &&
           a.element_bytes==b.element_bytes;
}
inline std::uint64_t payload_bytes(const Ticket& t) {
    if (!t.request || !t.epoch || t.rows<=0 || t.columns<=0 ||
        t.element_bytes<=0)
        throw std::invalid_argument("invalid transfer identity or dimensions");
    const auto r=std::uint64_t(t.rows), c=std::uint64_t(t.columns), b=std::uint64_t(t.element_bytes);
    if (r > std::numeric_limits<std::uint64_t>::max()/c ||
        r*c > std::numeric_limits<std::uint64_t>::max()/b)
        throw std::invalid_argument("overflowing transfer shape");
    return r*c*b;
}
inline std::uint32_t checksum(const std::vector<std::uint8_t>& v) {
    // FNV-1a is for this mock only, NOT cryptographic identity/authentication.
    std::uint32_t hash=2166136261u;
    for (std::uint8_t c : v) { hash ^= c; hash *= 16777619u; }
    return hash;
}
struct Message {
    Ticket ticket;
    std::vector<std::uint8_t> data;
    std::uint32_t crc;
};
class ConditioningMailbox {
    std::optional<Message> pending_;
public:
    void publish(Ticket t, std::vector<std::uint8_t> bytes) {
        if (pending_) throw std::invalid_argument("unconsumed conditioning data");
        if (payload_bytes(t)!=bytes.size()) throw std::invalid_argument("transfer size mismatch");
        const auto crc=checksum(bytes);
        pending_=Message{t,std::move(bytes),crc};
    }
    std::vector<std::uint8_t> receive(const Ticket& expected) {
        if (!pending_ || !same_ticket(pending_->ticket, expected))
            throw std::invalid_argument("stale or mismatched conditioning ticket");
        if (pending_->crc!=checksum(pending_->data))
            throw std::invalid_argument("conditioning checksum mismatch");
        auto output=std::move(pending_->data);
        pending_.reset();
        return output;
    }
    void cancel() { pending_.reset(); }
    bool pending() const { return pending_.has_value(); }
};

} // namespace bc_ltx_tp3_protocol
#endif
