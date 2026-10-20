#pragma once

// Owned-expert selection (heat, budget, balance) and publication. Included only by r3_moe_worker.cpp, at the position the code had in the single file (RCS03, pure move).

namespace {
struct Owned {
    std::vector<std::pair<uint32_t,uint32_t>> expert;
    uint64_t bytes = 0;
};

bool select_owned(const Container& f, r3_moe_gpu::Runner* gpu, uint64_t budget, uint32_t helper,
                  uint32_t helpers, Owned* out) {
    if (!out || !budget) {
        std::fprintf(stderr, "r3_moe_worker: no helper budget\n");
        return false;
    }
    if (!backend_accepts(f, gpu ? gpu->arch().c_str() : "gfx1100", false)) return false;
    if (gpu) {   // the contract's size must be the size the device layout really takes, or no budget arithmetic can be trusted
        const int64_t want = laid_out_bytes(f, gpu, 0, 0), got = gpu->expert_bytes(0, 0);
        if (want != got) {
            std::fprintf(stderr, "r3_moe_worker: expert size disagrees with the device layout (contract %lld, layout %lld)\n",
                         (long long)want, (long long)got);
            return false;
        }
    }
    std::vector<const RadFileProfile*> order;
    /* A container converted without calibration has no profile: take experts expert-major (unsorted)
     * so the budget spreads over all layers. */
    std::vector<RadFileProfile> flat;
    /* R3_MOE_WORKER_HEAT: route counts a profiling run wrote (R3_MOE_WORKER_HISTOGRAM). Coldest first, so the
     * engine keeps the experts most tokens route to and a decode step hands off as little as possible. */
    if (const char* hp = std::getenv("R3_MOE_WORKER_HEAT"); hp && *hp) {
        std::vector<uint32_t> heat((size_t)kMaxLayers * f.experts, 0);
        FILE* hf = std::fopen(hp, "rb");
        const size_t got = hf ? std::fread(heat.data(), sizeof(uint32_t), heat.size(), hf) : 0;
        if (hf) std::fclose(hf);
        if (got != heat.size()) {
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_HEAT %s: want %zu counts, read %zu\n",
                         hp, heat.size(), got);
            return false;
        }
        for (uint32_t l = 0; l < f.layers; ++l)
            for (uint32_t e = 0; e < f.experts; ++e)
                if (f.weight(l, e, 0))
                    flat.push_back(RadFileProfile{(int32_t)l, (int32_t)e, (float)heat[(size_t)l * f.experts + e], 0.0f});
        std::stable_sort(flat.begin(), flat.end(), [](const RadFileProfile& a, const RadFileProfile& b) {
            return a.share < b.share;
        });
        std::fprintf(stderr, "r3_moe_worker: owning the coldest experts of %s first\n", hp);
        for (const RadFileProfile& p : flat) order.push_back(&p);
    } else if (!f.profile || !f.h->prof_count) {
        std::fprintf(stderr, "r3_moe_worker: container has no expert profile; owning experts evenly across layers\n");
        /* Highest ids first: the engine's initial placement keeps low ids resident on the XTX and
         * leaves the high ids in its host pool (measured: every host-pool route was expert >= 302). */
        for (uint32_t e = f.experts; e-- > 0;)
            for (uint32_t l = 0; l < f.layers; ++l)
                if (f.weight(l, e, 0)) flat.push_back(RadFileProfile{(int32_t)l, (int32_t)e, 0.0f, 0.0f});
        for (const RadFileProfile& p : flat) order.push_back(&p);
    }
    for (uint64_t i = 0; flat.empty() && f.profile && i < f.h->prof_count; ++i) {
        const RadFileProfile& p = f.profile[i];
        if (p.layer < 0 || p.expert < 0 || (uint32_t)p.layer >= f.layers ||
            p.expert >= (int32_t)f.experts) continue;
        order.push_back(&p);
    }
    if (flat.empty()) std::stable_sort(order.begin(), order.end(), [](const RadFileProfile* a, const RadFileProfile* b) {
        if (a->share != b->share) return a->share < b->share;
        if (a->layer != b->layer) return a->layer < b->layer;
        return a->expert < b->expert;
    });
    /* R3_MOE_WORKER_BALANCE (RAD06, needs R3_MOE_WORKER_HEAT and two helpers): the same jointly owned set the
     * alternation below would pick, split per layer by predicted cost instead. A layer finishes when both helpers
     * have, so each layer wants max(helper0, helper1) small; alternating down one global ranking balances the number
     * of experts, not the work. Each helper runs this on its own, so everything here is a deterministic function of the
     * heat file and the budget (cost ties break on expert id). Cost of an expert, from its share of its layer's routes:
     *   lam = prefill_tokens * top_k * share      cost = (1 - exp(-lam)) * c0 + c1 * lam + (1 - exp(-lam_d)) * d0
     * (a fixed weight-tile part when the expert is active plus a part a routed row, and the same for a decode step).
     * The heat file holds max + 1 - count (so the hottest sort first); the counts are max - value.
     * R3_MOE_WORKER_BALANCE_SPEED="s0,s1" scales a helper's cost by 1/s (a slower card takes less). */
    if (const char* bv = std::getenv("R3_MOE_WORKER_BALANCE"); bv && *bv && std::strcmp(bv, "0") != 0) {
        if (helpers != 2 || flat.empty()) {
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_BALANCE needs R3_MOE_WORKER_HEAT and two helpers\n");
            return false;
        }
        struct Pick { uint32_t l, e; int64_t bytes; double cost; };
        float maxv = 0.0f;
        for (const RadFileProfile& hp : flat) maxv = std::max(maxv, hp.share);
        std::vector<double> layer_sum(f.layers, 0.0);
        for (const RadFileProfile& hp : flat) layer_sum[(uint32_t)hp.layer] += (double)(maxv - hp.share);
        const double tokens = 2048.0, decode = 11.0, c0 = 15.0, c1 = 0.09, d0 = 20.0;
        auto cost_of = [&](const RadFileProfile& hp) {
            const double share = layer_sum[(uint32_t)hp.layer] > 0.0 ? (double)(maxv - hp.share) / layer_sum[(uint32_t)hp.layer] : 0.0;
            const double lam = tokens * (double)kTopK * share, lam_d = decode * (double)kTopK * share;
            return (1.0 - std::exp(-lam)) * c0 + c1 * lam + (1.0 - std::exp(-lam_d)) * d0;
        };
        double speed[2] = {1.0, 1.0};
        if (const char* sv = std::getenv("R3_MOE_WORKER_BALANCE_SPEED"); sv && *sv) {
            double a = 0.0, b = 0.0;
            if (std::sscanf(sv, "%lf,%lf", &a, &b) == 2 && a > 0.0 && b > 0.0) { speed[0] = a; speed[1] = b; }
        }
        /* The jointly owned set: exactly the alternation's picks (both helpers' budgets filled in turn). */
        std::vector<Pick> joint;
        {
            std::vector<uint8_t> was_seen((size_t)f.layers * f.experts, 0);
            uint64_t held[2] = {0, 0}, pos = 0;
            for (const RadFileProfile* p : order) {
                const uint32_t l = (uint32_t)p->layer, e = (uint32_t)p->expert;
                uint8_t& w = was_seen[(size_t)l * f.experts + e];
                if (w) continue;
                w = 1;
                const uint32_t who = (uint32_t)(pos++ % 2);
                const int64_t bytes = laid_out_bytes(f, gpu, l, e);
                if (bytes <= 0) {
                    std::fprintf(stderr, "r3_moe_worker: expert (%u,%u) cannot be sized for the device\n", l, e);
                    return false;
                }
                if ((uint64_t)bytes > budget - held[who]) continue;
                held[who] += (uint64_t)bytes;
                joint.push_back(Pick{l, e, bytes, cost_of(*p)});
            }
        }
        /* Per layer, heaviest first onto the less loaded helper; a helper whose budget cannot take the expert passes it on.
         * The decision is made in layer / cost order, but each helper's list is emitted in the ranking order the
         * alternation would have used (hottest first): the order is also the order the weights are allocated in. */
        std::vector<uint32_t> idx(joint.size());
        for (uint32_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
            const Pick& x = joint[a];
            const Pick& y = joint[b];
            if (x.l != y.l) return x.l < y.l;
            if (x.cost != y.cost) return x.cost > y.cost;
            return x.e < y.e;
        });
        std::vector<int8_t> assigned(joint.size(), -1);
        uint64_t held[2] = {0, 0};
        double load[2] = {0.0, 0.0};
        uint32_t cur_layer = 0xffffffffu;
        for (uint32_t i : idx) {
            const Pick& pk = joint[i];
            if (pk.l != cur_layer) { cur_layer = pk.l; load[0] = load[1] = 0.0; }
            uint32_t who = load[0] + pk.cost / speed[0] <= load[1] + pk.cost / speed[1] ? 0u : 1u;
            if ((uint64_t)pk.bytes > budget - held[who]) who ^= 1u;
            if ((uint64_t)pk.bytes > budget - held[who]) continue;
            held[who] += (uint64_t)pk.bytes;
            load[who] += pk.cost / speed[who];
            assigned[i] = (int8_t)who;
        }
        for (uint32_t i = 0; i < joint.size(); ++i) {
            if (assigned[i] != (int8_t)helper) continue;
            out->expert.emplace_back(joint[i].l, joint[i].e);
            out->bytes += (uint64_t)joint[i].bytes;
        }
        std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_BALANCE: per-layer cost split of %zu jointly owned experts\n",
                     joint.size());
        if (out->expert.empty()) {
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_MIB fits no profiled expert\n");
            return false;
        }
        return true;
    }
    std::vector<uint8_t> seen((size_t)f.layers * f.experts, 0);
    uint64_t position = 0;
    for (const RadFileProfile* p : order) {
        const uint32_t l = (uint32_t)p->layer, e = (uint32_t)p->expert;
        uint8_t& was = seen[(size_t)l * f.experts + e];
        if (was) continue;
        was = 1;
        /* Helper h of n takes every n-th candidate: disjoint sets, each spread over the same order. */
        if (position++ % helpers != helper) continue;
        const int64_t bytes = laid_out_bytes(f, gpu, l, e);
        if (bytes <= 0) {
            std::fprintf(stderr, "r3_moe_worker: expert (%u,%u) cannot be sized for the device\n", l, e);
            return false;
        }
        if ((uint64_t)bytes > budget - out->bytes) continue;
        out->expert.emplace_back(l, e);
        out->bytes += (uint64_t)bytes;
    }
    if (out->expert.empty()) {
        std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_MIB fits no profiled expert\n");
        return false;
    }
    return true;
}

void publish_owned(void* map, const Owned& owned, uint32_t helper) {
    Header* h = static_cast<Header*>(map);
    __atomic_fetch_add(&h->owned_bytes, owned.bytes, __ATOMIC_ACQ_REL);
    for (auto [l, e] : owned.expert) {
        if (!set_owned(map, l, e, helper))
            std::fprintf(stderr, "r3_moe_worker: duplicate/invalid ownership (%u,%u)\n", l, e);
    }
}

bool parse_u32(const char* s, uint32_t lo, uint32_t hi, uint32_t* out) {
    if (!s || !*s || !out) return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long v = std::strtoul(s, &end, 10);
    if (errno || !end || *end || v < lo || v > hi) return false;
    *out = (uint32_t)v;
    return true;
}
} // namespace
