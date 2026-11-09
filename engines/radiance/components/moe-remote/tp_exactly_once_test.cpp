// RAD49 (PR #771 review): a card-free reference of the full-width-once TP sum, end to end.
//
// Two engine ranks get the same routed expert ids (the router is replicated). For every layer each rank takes each route's
// role from route_role (the publish kernel's own definition) and the owner bitmaps through r3_moe_owner_has (the kernel's
// predicate): it masks the helper-owned routes out of its local GEMMs, and only the layer's consumer (consumer_of, alternating)
// sends them to the owning helper's slot (slot_of). Each helper computes its routes at full width; the consumer adds those rows,
// the other rank adds nothing for them. Both ranks add their width slice of the locally resident experts and of the shared
// expert. The "all-reduce" sums the two ranks. Small integer weights keep every float sum exact, so the TP result must equal the
// one-device reference bit for bit; a mutant that drops the non-consumer's mask, or lets both ranks add the helper rows, must
// not. The heat histogram is checked too: both ranks count every route of every layer, so rank 0's histogram covers the layers
// rank 1 consumes.
//
// The fused path (RAD49, moe_mailbox_inject): where the engine's fused MoE gather + reducing hc write takes the gather, the
// block gathers nothing; instead the consumer gives each helper row of a token a spare sorted row past the scatter's live
// count, at that helper's first masked slot of the token (inject_slot), with weight 1 (kInjectWeightBf16). The fused gather
// then sums every claimed slot of the token in slot order through the inverse table, adds the rank's shared slice, and the
// reduction sums the ranks. That too must equal the reference bit for bit, with every injected row on a spare sorted row;
// mutants that inject on both ranks, keep the route's router weight, or put two helpers' rows at one slot must not.
#include "moe_remote_hook.h"
#include "moe_remote_shm.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace r3::moe_shm;

namespace {
void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "tp_exactly_once_test: FAILED %s\n", why); std::exit(1); }
}

constexpr uint32_t kL = 4, kE = 8, kH = 6, kI = 4, kT = 5, kK = 3, kWorld = 2, kHelpers = 2;
using Row = std::array<float, kH>;

/* Expert e of layer l: out[c] = sum_j D[c][j] * (sum_x G[j][x] * in[x]), linear with small integer weights (exact in float).
 * A TP rank holds the intermediate columns j with j % kWorld == rank (its width slice); a helper holds all of them. */
float G(uint32_t l, uint32_t e, uint32_t j, uint32_t x) { return float(int((l * 7 + e * 5 + j * 3 + x) % 5) - 2); }
float D(uint32_t l, uint32_t e, uint32_t c, uint32_t j) { return float(int((l * 3 + e * 11 + c * 2 + j * 5) % 7) - 3); }
constexpr uint32_t kShared = kE;   /* the shared expert, split by width like every resident expert */

Row expert(uint32_t l, uint32_t e, const Row& in, int slice /* -1: full width */) {
    Row out{};
    for (uint32_t j = 0; j < kI; ++j) {
        if (slice >= 0 && j % kWorld != (uint32_t)slice) continue;
        float mid = 0.0f;
        for (uint32_t x = 0; x < kH; ++x) mid += G(l, e, j, x) * in[x];
        for (uint32_t c = 0; c < kH; ++c) out[c] += D(l, e, c, j) * mid;
    }
    return out;
}
void add(Row& a, const Row& b, float w) { for (uint32_t c = 0; c < kH; ++c) a[c] += w * b[c]; }

struct Model {
    std::array<std::array<Row, kT>, kL> in{};
    std::array<std::array<std::array<int32_t, kK>, kT>, kL> ids{};
    std::array<std::array<std::array<float, kK>, kT>, kL> w{};
    std::vector<uint8_t> owner_bits;   /* helper-strided, layer-major bitmaps as the hook uploads them */
    uint32_t bitmap_bytes = 0;
    int owner_of(uint32_t l, uint32_t e) const {
        for (uint32_t k = 0; k < kHelpers; ++k)
            if (r3_moe_owner_has(owner_bits.data() + k * bitmap_bytes, l, e, kE)) return int(k);
        return -1;
    }
};

Model make_model() {
    Model m;
    m.bitmap_bytes = (kL * kE + 7) / 8;
    m.owner_bits.assign(kHelpers * m.bitmap_bytes, 0);
    for (uint32_t l = 0; l < kL; ++l)
        for (uint32_t e = 0; e < kE; ++e) {
            const uint32_t pick = (e + l) % 4;   /* 0: helper 0, 1: helper 1, else engine-resident */
            if (pick < kHelpers) {
                const uint32_t bit = l * kE + e;
                m.owner_bits[pick * m.bitmap_bytes + bit / 8] |= uint8_t(1u << (bit % 8));
            }
        }
    for (uint32_t l = 0; l < kL; ++l)
        for (uint32_t t = 0; t < kT; ++t) {
            for (uint32_t c = 0; c < kH; ++c) m.in[l][t][c] = float(int((l + 3 * t + 5 * c) % 5) - 2);
            for (uint32_t s = 0; s < kK; ++s) {
                m.ids[l][t][s] = int32_t((l * 3 + t * 2 + s * 3) % kE);
                m.w[l][t][s] = float(1 + (t + s + l) % 3);
            }
        }
    return m;
}

struct Mutation {
    bool skip_mask_on_non_consumer = false, both_ranks_add_helper_rows = false;
    /* fused-path mutants */
    bool keep_route_weight = false, one_slot_for_all_helpers = false;
};

/* One layer on one rank: its partial output and the routes it sent to each slot. */
struct RankOut {
    std::array<Row, kT> partial{};
    std::array<std::vector<std::pair<uint32_t, uint32_t>>, kMaxSlots> sent;   /* (token, top-k slot) */
    std::array<uint32_t, kE> heat{};
};

/* Route roles for one layer on one rank: the ids with the masked routes cleared, and the routes sent to each slot. */
void take_roles(const Model& m, uint32_t l, uint32_t rank, Mutation mut,
                std::array<std::array<int32_t, kK>, kT>& ids, RankOut& r) {
    ids = m.ids[l];   /* every rank gets the same routed ids */
    for (uint32_t t = 0; t < kT; ++t)
        for (uint32_t s = 0; s < kK; ++s) {
            const int32_t e = ids[t][s];
            const int owner = m.owner_of(l, uint32_t(e));
            const RouteRole role = route_role(consumer_of(l, kWorld) == rank, true, owner >= 0, false);
            if (role.count_heat) ++r.heat[uint32_t(e)];
            if (role.publish || (mut.both_ranks_add_helper_rows && owner >= 0))
                r.sent[slot_of(uint32_t(owner), rank, kWorld)].emplace_back(t, s);
            const bool consumer = consumer_of(l, kWorld) == rank;
            if (role.mask_local && !(mut.skip_mask_on_non_consumer && !consumer)) ids[t][s] = -1;
        }
    for (uint32_t slot = 0; slot < kMaxSlots; ++slot)
        for (auto [t, s] : r.sent[slot]) {
            check(slot_consumer(slot, kWorld) == rank, "a rank only sends to its own slots");
            check(m.owner_of(l, uint32_t(m.ids[l][t][s])) == int(slot_helper(slot, kWorld)),
                  "a route goes to the helper that owns it");
        }
}

RankOut run_rank(const Model& m, uint32_t l, uint32_t rank, Mutation mut) {
    RankOut r;
    std::array<std::array<int32_t, kK>, kT> ids{};
    take_roles(m, l, rank, mut, ids, r);
    for (uint32_t t = 0; t < kT; ++t) {
        /* local resident experts (width slice) in slot order, then this rank's slice of the shared expert */
        for (uint32_t s = 0; s < kK; ++s)
            if (ids[t][s] >= 0) add(r.partial[t], expert(l, uint32_t(ids[t][s]), m.in[l][t], int(rank)), m.w[l][t][s]);
        add(r.partial[t], expert(l, kShared, m.in[l][t], int(rank)), 1.0f);
    }
    /* the helpers: each computes its slot's routes at full width; the rank that published adds the rows */
    for (uint32_t slot = 0; slot < kMaxSlots; ++slot)
        for (auto [t, s] : r.sent[slot])
            add(r.partial[t], expert(l, uint32_t(m.ids[l][t][s]), m.in[l][t], -1), m.w[l][t][s]);
    return r;
}

bool same(const std::array<Row, kT>& a, const std::array<Row, kT>& b);

float bf16_value(uint16_t b) {
    const uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

/* One layer on one rank, fused: the scatter's sorted rows (local routes grouped by expert, then -1), the down rows (each
 * sorted row's unweighted width-slice output), the router weights; the inject (the consumer's helper rows onto spare sorted
 * rows, weight 1); then the fused gather (every claimed slot of a token in slot order, through the inverse table) and the
 * rank's shared slice. Counts the injected rows. */
RankOut run_rank_fused(const Model& m, uint32_t l, uint32_t rank, Mutation mut, uint32_t* injected) {
    RankOut r;
    std::array<std::array<int32_t, kK>, kT> ids{};
    take_roles(m, l, rank, mut, ids, r);
    constexpr uint32_t cap = kT * kK;
    std::array<int32_t, cap> sorted{};
    sorted.fill(-1);
    std::array<Row, cap> down{};
    std::array<float, cap> ew{};
    uint32_t live = 0;
    for (uint32_t e = 0; e < kE; ++e)   /* the scatter: grouped by expert; eoff[n_expert] == live */
        for (uint32_t t = 0; t < kT; ++t)
            for (uint32_t s = 0; s < kK; ++s)
                if (ids[t][s] == int32_t(e)) {
                    sorted[live] = int32_t(t * kK + s);
                    down[live] = expert(l, e, m.in[l][t], int(rank));
                    ++live;
                }
    for (uint32_t t = 0; t < kT; ++t)
        for (uint32_t s = 0; s < kK; ++s) ew[t * kK + s] = m.w[l][t][s];
    /* the helpers' planes: per helper, per token, the weighted sum of its routes at full width; the route mark */
    std::array<std::array<Row, kT>, kHelpers> plane{};
    std::array<std::array<bool, kT>, kHelpers> has{};
    std::array<uint32_t, cap> rmask{};
    bool live_slots = false;
    for (uint32_t k = 0; k < kHelpers; ++k)
        for (auto [t, s] : r.sent[slot_of(k, rank, kWorld)]) {
            add(plane[k][t], expert(l, uint32_t(m.ids[l][t][s]), m.in[l][t], -1), m.w[l][t][s]);
            has[k][t] = true;
            check(rmask[t * kK + s] == 0, "one route a slot");
            rmask[t * kK + s] = k + 1;
            live_slots = true;
        }
    if (!mut.both_ranks_add_helper_rows)
        check(!live_slots || consumer_of(l, kWorld) == rank, "only the consumer finds live helper slots (the kernel traps)");
    /* the inject */
    uint32_t next = live;
    for (uint32_t t = 0; t < kT; ++t)
        for (uint32_t k = 0; k < kHelpers; ++k) {
            int32_t s = inject_slot(rmask.data() + t * kK, kK, k);
            if (mut.one_slot_for_all_helpers && s >= 0)
                for (uint32_t j = 0; j < kK; ++j)
                    if (rmask[t * kK + j]) { s = int32_t(j); break; }
            check(has[k][t] == (s >= 0), "a helper has a row for a token exactly when it owns one of its slots");
            if (s < 0) continue;
            const uint32_t p = next++;
            check(p < cap, "every injected row finds a spare sorted row");
            check(sorted[p] == -1, "an injected row takes a spare sorted row");
            const uint32_t flat = t * kK + uint32_t(s);
            sorted[p] = int32_t(flat);
            if (!mut.keep_route_weight) ew[flat] = bf16_value(kInjectWeightBf16);
            down[p] = plane[k][t];
            ++*injected;
        }
    /* the fused gather: the inverse table (the last sorted row claiming a slot wins, as r4d_ar_gather_inv_kernel) */
    std::array<int32_t, cap> inv{};
    inv.fill(-1);
    for (uint32_t i = 0; i < cap; ++i)
        if (sorted[i] >= 0) inv[uint32_t(sorted[i])] = int32_t(i);
    for (uint32_t t = 0; t < kT; ++t) {
        for (uint32_t s = 0; s < kK; ++s) {
            const int32_t i = inv[t * kK + s];
            if (i >= 0) add(r.partial[t], down[uint32_t(i)], ew[t * kK + s]);
        }
        add(r.partial[t], expert(l, kShared, m.in[l][t], int(rank)), 1.0f);
    }
    return r;
}

/* The fused TP sum over every layer; true when it equals the one-device reference exactly. Counts the injected rows. */
bool tp_fused_matches(const Model& m, Mutation mut, uint32_t* injected) {
    bool all = true;
    for (uint32_t l = 0; l < kL; ++l) {
        std::array<Row, kT> ref{};
        for (uint32_t t = 0; t < kT; ++t) {
            for (uint32_t s = 0; s < kK; ++s) add(ref[t], expert(l, uint32_t(m.ids[l][t][s]), m.in[l][t], -1), m.w[l][t][s]);
            add(ref[t], expert(l, kShared, m.in[l][t], -1), 1.0f);
        }
        const RankOut r0 = run_rank_fused(m, l, 0, mut, injected), r1 = run_rank_fused(m, l, 1, mut, injected);
        std::array<Row, kT> sum = r0.partial;
        for (uint32_t t = 0; t < kT; ++t) add(sum[t], r1.partial[t], 1.0f);   /* the reducing write */
        all = all && same(sum, ref);
    }
    return all;
}

bool same(const std::array<Row, kT>& a, const std::array<Row, kT>& b) {
    return std::memcmp(a.data(), b.data(), sizeof(a)) == 0;
}

/* The TP sum over every layer; true when it equals the one-device reference exactly. Fills the per-rank heat. */
bool tp_matches(const Model& m, Mutation mut, std::array<std::array<uint32_t, kE>, kL>* heat0,
                std::array<std::array<uint32_t, kE>, kL>* heat1) {
    bool all = true;
    for (uint32_t l = 0; l < kL; ++l) {
        std::array<Row, kT> ref{};
        for (uint32_t t = 0; t < kT; ++t) {
            for (uint32_t s = 0; s < kK; ++s) add(ref[t], expert(l, uint32_t(m.ids[l][t][s]), m.in[l][t], -1), m.w[l][t][s]);
            add(ref[t], expert(l, kShared, m.in[l][t], -1), 1.0f);
        }
        const RankOut r0 = run_rank(m, l, 0, mut), r1 = run_rank(m, l, 1, mut);
        std::array<Row, kT> sum = r0.partial;
        for (uint32_t t = 0; t < kT; ++t) add(sum[t], r1.partial[t], 1.0f);   /* the all-reduce */
        all = all && same(sum, ref);
        if (!mut.skip_mask_on_non_consumer && !mut.both_ranks_add_helper_rows) {
            const RankOut& non = consumer_of(l, kWorld) == 0 ? r1 : r0;
            for (const auto& v : non.sent) check(v.empty(), "the non-consumer publishes nothing");
            size_t published = 0, owned = 0;
            for (const auto& v : (consumer_of(l, kWorld) == 0 ? r0 : r1).sent) published += v.size();
            for (uint32_t t = 0; t < kT; ++t)
                for (uint32_t s = 0; s < kK; ++s) owned += m.owner_of(l, uint32_t(m.ids[l][t][s])) >= 0;
            check(published == owned && owned > 0, "every helper-owned route is published exactly once");
        }
        if (heat0) (*heat0)[l] = r0.heat;
        if (heat1) (*heat1)[l] = r1.heat;
    }
    return all;
}
}  // namespace

int main() {
    const Model m = make_model();
    {   /* the model exercises local, helper-0, helper-1 and shared contributions on both consumers' layers */
        bool h0[2]{}, h1[2]{}, local[2]{};
        for (uint32_t l = 0; l < kL; ++l)
            for (uint32_t t = 0; t < kT; ++t)
                for (uint32_t s = 0; s < kK; ++s) {
                    const int o = m.owner_of(l, uint32_t(m.ids[l][t][s]));
                    const uint32_t c = consumer_of(l, kWorld);
                    (o == 0 ? h0 : o == 1 ? h1 : local)[c] = true;
                }
        check(h0[0] && h0[1] && h1[0] && h1[1] && local[0] && local[1], "every contribution on both consumers' layers");
    }
    std::array<std::array<uint32_t, kE>, kL> heat0{}, heat1{};
    check(tp_matches(m, {}, &heat0, &heat1), "the TP2 sum with two helpers equals the one-device reference exactly");
    /* heat coverage: rank 0 counts every route of every layer, the odd ones rank 1 consumes included */
    for (uint32_t l = 0; l < kL; ++l) {
        std::array<uint32_t, kE> want{};
        for (uint32_t t = 0; t < kT; ++t)
            for (uint32_t s = 0; s < kK; ++s) ++want[uint32_t(m.ids[l][t][s])];
        check(heat0[l] == want, "rank 0's histogram counts every route of the layer, consumed by it or not");
        check(heat1[l] == heat0[l], "both ranks count the same routes (replicated router)");
        if (consumer_of(l, kWorld) == 1) {
            uint32_t n = 0;
            for (uint32_t v : heat0[l]) n += v;
            check(n == kT * kK, "an odd layer (rank 1 consumes it) is fully in rank 0's histogram");
        }
    }
    Mutation no_mask; no_mask.skip_mask_on_non_consumer = true;
    check(!tp_matches(m, no_mask, nullptr, nullptr), "a non-consumer that keeps helper routes local double counts them");
    Mutation both; both.both_ranks_add_helper_rows = true;
    check(!tp_matches(m, both, nullptr, nullptr), "both ranks adding full-width helper rows double counts them");

    /* the fused gather + reducing write */
    check(bf16_value(kInjectWeightBf16) == 1.0f, "the injected weight is exactly 1");
    {   /* the model has a token with rows from both helpers on one rank (two injections for one token) */
        bool two = false;
        for (uint32_t l = 0; l < kL; ++l)
            for (uint32_t t = 0; t < kT; ++t) {
                bool h[kHelpers]{};
                for (uint32_t s = 0; s < kK; ++s) {
                    const int o = m.owner_of(l, uint32_t(m.ids[l][t][s]));
                    if (o >= 0) h[o] = true;
                }
                two = two || (h[0] && h[1]);
            }
        check(two, "some token has rows from both helpers");
    }
    uint32_t injected = 0;
    check(tp_fused_matches(m, {}, &injected), "the fused TP2 gather with injected helper rows equals the reference exactly");
    {   /* one injected row per (layer, token, helper with a route of that token) */
        uint32_t want = 0;
        for (uint32_t l = 0; l < kL; ++l)
            for (uint32_t t = 0; t < kT; ++t)
                for (uint32_t k = 0; k < kHelpers; ++k) {
                    bool any = false;
                    for (uint32_t s = 0; s < kK; ++s) any = any || m.owner_of(l, uint32_t(m.ids[l][t][s])) == int(k);
                    want += any;
                }
        check(injected == want && want > 0, "each helper row is injected exactly once");
    }
    uint32_t sink = 0;
    check(!tp_fused_matches(m, both, &sink), "injecting the helper rows on both ranks double counts them");
    Mutation weight; weight.keep_route_weight = true;
    check(!tp_fused_matches(m, weight, &sink), "an injected row keeping its route's router weight is weighted twice");
    Mutation one_slot; one_slot.one_slot_for_all_helpers = true;
    check(!tp_fused_matches(m, one_slot, &sink), "two helpers' rows at one slot lose one of them");
    /* a one-rank engine is its own consumer for every layer and publishes every helper route */
    for (uint32_t l = 0; l < kL; ++l) {
        const RouteRole r = route_role(consumer_of(l, 1) == 0, true, true, false);
        check(r.publish && r.mask_local && r.count_heat, "a one-rank engine publishes and masks every owned route");
    }
    check(!route_role(true, true, true, true).mask_local && route_role(true, true, true, true).publish,
          "the shadow diagnostic keeps the local path but still publishes");
    check(!route_role(true, false, true, false).count_heat && !route_role(true, false, true, false).publish,
          "an out-of-range id is neither counted nor published");
    check(!route_role(false, true, true, false).publish && route_role(false, true, true, false).mask_local,
          "a non-consumer masks but does not publish");
    std::puts("tp_exactly_once_test: ok");
    return 0;
}
