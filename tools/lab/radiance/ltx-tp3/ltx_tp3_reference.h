// Card-free reference only. No engine call sites, HIP device code or production registration.
// Mirrors LTX2's all-head RMS and fused head slicing semantics for the TP3 migration.
// Read tools/lab/radiance/ltx-tp3/README.md before reusing this in an engine patch.
#ifndef BC_LTX_TP3_REFERENCE_H
#define BC_LTX_TP3_REFERENCE_H

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bc_ltx_tp3_reference {

constexpr int kRanks = 3;
constexpr int kVideoHeads = 32;
constexpr int kVideoHeadDim = 128;
constexpr int kAudioHeads = 32;
constexpr int kAudioHeadDim = 64;
constexpr int kIq8rTransform = 128;
constexpr int kIq8rGroup = 256;

using Shares = std::array<int, kRanks>;

struct Slice {
    int offset;
    int width;
};

inline std::array<Slice, kRanks> split(int total, Shares shares, int granularity) {
    if (total <= 0 || granularity <= 0 || total % granularity)
        throw std::invalid_argument("invalid width or granularity");
    int sum = 0;
    for (int s : shares) {
        if (s <= 0 || s % granularity)
            throw std::invalid_argument("empty or misaligned shard");
        if (s > total - sum)
            throw std::invalid_argument("shard exceeds width");
        sum += s;
    }
    if (sum != total) throw std::invalid_argument("shards do not cover width");
    std::array<Slice, kRanks> out{};
    int at = 0;
    for (int rank = 0; rank < kRanks; ++rank) {
        out[rank] = {at, shares[rank]};
        at += shares[rank];
    }
    return out;
}

// A caller specifies whole heads, never fractions of a head.
// This also works for the audio-driven cross-modal attention (32 x 64).
inline std::array<Slice, kRanks> attention_slices(int heads, int head_dim, Shares head_shares) {
    if (heads <= 0 || head_dim <= 0 || int64_t(heads)*head_dim > std::numeric_limits<int>::max())
        throw std::invalid_argument("bad attention geometry");
    for (int& s : head_shares) {
        if (s <= 0 || int64_t(s)*head_dim > std::numeric_limits<int>::max())
            throw std::invalid_argument("invalid or overflowing head share");
        s *= head_dim;
    }
    return split(heads * head_dim, head_shares, head_dim);
}

struct FusedPiece {
    int source_row;       // row in the full [q | k | v] checkpoint projection
    int local_row;        // row in this rank's compact [q | k | v] projection
    int count;
};

// LTX self-attention stores all Q rows, then K, then V. A contiguous
// three-way split of the *fused tensor* is wrong: each rank needs all 3 banks.
inline std::array<FusedPiece, 3> fused_qkv_pieces(
        int total_inner, Slice owned) {
    if (total_inner <= 0 || total_inner > std::numeric_limits<int>::max()/3 ||
        owned.offset < 0 || owned.width <= 0 ||
        int64_t(owned.offset) + owned.width > total_inner)
        throw std::invalid_argument("bad fused projection ownership");
    return {{
        {owned.offset, 0, owned.width},
        {total_inner + owned.offset, owned.width, owned.width},
        {2 * total_inner + owned.offset, 2 * owned.width, owned.width},
    }};
}

// IQ8R rotates activations in K blocks of 128, but groups scale factors
// over 256 K entries. K cuts must preserve both boundaries. Output-row
// slicing has no K cut and can retain the original row encodings unchanged.
inline std::array<Slice, kRanks> iq8r_input_slices(int full_k, Shares widths) {
    if (full_k % kIq8rTransform || full_k % kIq8rGroup)
        throw std::invalid_argument("unrepresentable IQ8R K extent");
    return split(full_k, widths, kIq8rGroup);
}

inline std::array<double, kRanks> local_square_sums(
        const std::array<std::vector<float>, kRanks>& shards) {
    std::array<double, kRanks> sums{};
    for (int rank = 0; rank < kRanks; ++rank) {
        // f32 kernel reduction order is a later parity gate; double here
        // isolates algebraic correctness from native HIP summation order.
        for (float x : shards[rank]) sums[rank] += double(x) * double(x);
    }
    return sums;
}

// All Q/K heads share one RMS denominator, not one denominator per TP rank.
// Simulates the collective (rank order is deterministic) and local scale.
inline std::array<std::vector<float>, kRanks> global_rms(
        const std::array<std::vector<float>, kRanks>& shards,
        const std::array<std::vector<float>, kRanks>& gains, double eps) {
    if (!std::isfinite(eps) || eps <= 0.0)
        throw std::invalid_argument("invalid RMS epsilon");
    std::size_t count = 0;
    for (int r = 0; r < kRanks; ++r) {
        if (shards[r].empty() || gains[r].size() != shards[r].size())
            throw std::invalid_argument("missing shard or gain");
        count += shards[r].size();
        for (float x : shards[r])
            if (!std::isfinite(x)) throw std::invalid_argument("nonfinite activation");
        for (float x : gains[r])
            if (!std::isfinite(x)) throw std::invalid_argument("nonfinite gain");
    }
    const auto sums = local_square_sums(shards);
    double global = 0;
    for (int r = 0; r < kRanks; ++r) global += sums[r];
    const double factor = 1.0 / std::sqrt(global / double(count) + eps);
    std::array<std::vector<float>, kRanks> output;
    for (int r = 0; r < kRanks; ++r) {
        output[r].reserve(shards[r].size());
        for (std::size_t i = 0; i < shards[r].size(); ++i)
            output[r].push_back(float(double(shards[r][i]) * factor * gains[r][i]));
    }
    return output;
}

enum class Arch { gfx1100, gfx1201, gfx1030 };
struct Card {
    int id;                 // test-only physical card identity; production uses verified BDFs
    Arch arch;
    int vram_mib;
    int reserved_mib;
};

struct FourCardPlan {
    std::array<Card, kRanks> dit;
    Card text;
    int text_weights_mib;
    int text_workspace_mib; // must be measured; never assume zero
    int min_free_mib;
};

// This is a mock admission model, NOT a replacement for native measured
// resource admission. It deliberately refuses unknown or overfull budgets.
inline void validate_four_card_plan(const FourCardPlan& p) {
    if (p.dit[0].arch != Arch::gfx1100 ||
        p.dit[1].arch != Arch::gfx1100 ||
        p.dit[2].arch != Arch::gfx1201 ||
        p.text.arch != Arch::gfx1030)
        throw std::invalid_argument("unexpected TP3 / text architecture");
    for (int i = 0; i < kRanks; ++i) {
        if (p.dit[i].id == p.text.id)
            throw std::invalid_argument("text card overlaps a TP rank");
        if (p.dit[i].vram_mib <= p.dit[i].reserved_mib ||
            p.dit[i].reserved_mib < 0)
            throw std::invalid_argument("DiT card has no memory headroom");
        for (int j = 0; j < i; ++j)
            if (p.dit[i].id == p.dit[j].id)
                throw std::invalid_argument("duplicate DiT card");
    }
    if (p.text_weights_mib <= 0 || p.text_workspace_mib <= 0 ||
        p.min_free_mib <= 0 || p.text.reserved_mib < 0 ||
        p.text.vram_mib <= p.text.reserved_mib)
        throw std::invalid_argument("invalid text memory estimate");
    const int64_t used = int64_t(p.text.reserved_mib) + p.text_weights_mib +
                         p.text_workspace_mib + p.min_free_mib;
    if (used > p.text.vram_mib)
        throw std::invalid_argument("text card cannot hold weights and working set");
}

// No wrapped byte extents in mock admission.
inline int64_t checked_product(int64_t a,int64_t b) {
    if (a<=0 || b<=0 || a>std::numeric_limits<int64_t>::max()/b)
        throw std::invalid_argument("invalid or overflowing extent");
    return a*b;
}
// LTX2 Gemma stores every hidden layer plus the embedding: the connector
// input is [T, C * (L + 1)] bf16, not the much smaller connector output.
// If connectors execute on the fast ranks, this *stack* must move from gfx1030.
inline int64_t gemma_stack_bytes(int rows_per_prompt, int prompts,
                                 int hidden, int layers, int dtype_bytes) {
    if (rows_per_prompt <= 0 || prompts <= 0 || hidden <= 0 ||
        layers <= 0 || dtype_bytes <= 0)
        throw std::invalid_argument("invalid Gemma stack shape");
    return checked_product(checked_product(checked_product(
        checked_product(rows_per_prompt,prompts),hidden),int64_t(layers)+1),dtype_bytes);
}

// Both prompts' video/audio connector rows, as declared by LTX2. The
// model still needs a real native handoff and replay/step synchronization.
inline int64_t conditioning_bytes(int rows_per_prompt, int prompts,
                                   int video_width, int audio_width, int dtype_bytes) {
    if (rows_per_prompt <= 0 || prompts <= 0 || video_width <= 0 ||
        audio_width <= 0 || dtype_bytes <= 0)
        throw std::invalid_argument("invalid text-conditioning geometry");
    return checked_product(checked_product(
        checked_product(rows_per_prompt,prompts),int64_t(video_width)+audio_width),dtype_bytes);
}

} // namespace bc_ltx_tp3_reference
#endif
