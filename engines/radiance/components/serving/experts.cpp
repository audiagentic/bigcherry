#include "experts.h"
#include "worker_share.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <set>

namespace bc::experts {
using r3::moe_shm::kMaxLayers;
using r3::moe_shm::kTopK;

int64_t canonical_expert_bytes(const Container& f, uint32_t l, uint32_t e) {
    int64_t n = 0;
    for (uint32_t slot = 0; slot < 2; ++slot) {
        const RadFileEntry* w = f.weight(l, e, slot);
        if (!w) return -1;
        for (uint32_t p = 0; p < w->n_planes; ++p) {
            const uint64_t b = f.plane(w, (int)p)->bytes;
            if (b > (uint64_t)std::numeric_limits<int64_t>::max() - (uint64_t)n) return -1;
            n += (int64_t)b;
        }
    }
    return n;
}

/* What the container says an expert is (RCS09/RAD11): its weight format, scale type and tile, read from the encoding, never assumed.
 * W4NL64 = 4-bit codes through a lookup table with scales per 64 columns; FP8_E4M3_BLOCK = E4M3 codes with a scale per tile.
 * Anything else is MOE_FMT_NONE and is refused by name before any expert is laid out. */
static uint32_t scale_dtype_of(uint32_t dtype) {
    switch (dtype) {
        case RAD_F8E4M3: return MOE_SCALE_E4M3;
        case RAD_F16: return MOE_SCALE_F16;
        case RAD_BF16: return MOE_SCALE_BF16;
        case RAD_F32: return MOE_SCALE_F32;
        default: return MOE_SCALE_NONE;
    }
}

bool describe_expert(const Container& f, uint32_t l, uint32_t e, MoeExpertDesc* out) {
    const RadFileEntry* gu = f.weight(l, e, 0);
    const RadFileEntry* dn = f.weight(l, e, 1);
    if (!gu || !dn || !out) return false;
    const RadEncoding* enc = f.encoding(gu);
    if (!enc || !rad_enc_valid(enc)) return false;
    MoeExpertDesc d{};
    const int codes = rad_enc_find(enc, "codes");
    const int table = rad_enc_find(enc, "table");
    const int scale0 = rad_enc_find_level(enc, "scale", 0);
    const int scale1 = rad_enc_find_level(enc, "scale", 1);
    if (codes >= 0 && scale0 >= 0) {
        const RadEncPlane& cp = enc->plane[codes];
        const RadEncPlane& sp = enc->plane[scale0];
        if (rad_dtype_bits(cp.dtype) == 4 && table >= 0) {
            /* the two scale levels relayout to one E4M3 byte per 64 columns (the helper checks this geometry at upload) */
            (void)scale1;
            d.weight = MoeWeightDesc{MOE_FMT_W4NL64, MOE_ACT_FP8_E4M3_GROUP, MOE_SCALE_E4M3, 0, (uint32_t)sp.block[1], 0};
        } else if (cp.dtype == RAD_F8E4M3 && table < 0) {
            d.weight = MoeWeightDesc{MOE_FMT_FP8_E4M3_BLOCK, MOE_ACT_FP8_E4M3_GROUP, scale_dtype_of(sp.dtype),
                                     (uint32_t)sp.block[0], (uint32_t)sp.block[1], 0};
        }
    }
    int64_t gr = 1, gc = 1, dr = 1, dc = 1;
    rad_enc_view(gu->rank, gu->shape, &gr, &gc);
    rad_enc_view(dn->rank, dn->shape, &dr, &dc);
    d.gu_rows = (uint64_t)gr; d.gu_cols = (uint64_t)gc;
    d.dn_rows = (uint64_t)dr; d.dn_cols = (uint64_t)dc;
    *out = d;
    return true;
}

/* The shape the container implies (RCS09): layers, routed experts, hidden, expert intermediate width. top_k is the engine's. */
bool container_shape(const Container& f, MoeShape* s) {
    if (!f.experts || !f.hidden) return false;
    *s = MoeShape{f.layers, f.experts, f.hidden, f.top_k, f.intermediate, 0, 0};
    return true;
}


/* The backend of the card this process runs on, from R3_MOE_WORKER_FORMATS (the weight formats its kernels run; engines/radiance/archs.toml,
 * set by the launcher and the harness). A card type is data: nothing here knows which architectures exist. */
bool helper_backend(const char* arch, const char* formats, r3::moe_backend::Backend* out) {
    uint32_t mask = 0;
    const char* env = formats;
    if (!r3::moe_backend::parse_formats(env, &mask)) {
        std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_FORMATS must list the weight formats this card runs (w4nl64, fp8_e4m3_block), got '%s'; "
                             "the card types are engines/radiance/archs.toml\n", env ? env : "");
        return false;
    }
    *out = r3::moe_backend::Backend{};
    std::strncpy(out->name, arch, sizeof(out->name) - 1);
    out->formats = mask;
    return true;
}

/* Whether this card's backend can run the container's experts, and why not when it cannot. */
bool backend_accepts(const Container& f, const char* arch, const char* formats, bool verbose) {
    MoeShape shape{};
    MoeExpertDesc d{};
    if (!container_shape(f, &shape) || !describe_expert(f, 0, 0, &d)) {
        std::fprintf(stderr, "r3_moe_worker: the container's first expert could not be described\n");
        return false;
    }
    shape.top_k = 1;      // the container does not carry the engine's top-k; the check here is the format and geometry
    r3::moe_backend::Backend backend{};
    if (!helper_backend(arch, formats, &backend)) return false;
    const int rc = r3::moe_backend::probe(backend, shape, d.weight);
    if (verbose || rc != MOE_OK)
        std::fprintf(stderr,
                     "r3_moe_worker: container: %u layers, %u experts, hidden %u, expert width %u, weight format %s "
                     "(scale dtype %u, tile %ux%u), %s backend: %s\n",
                     shape.layers, shape.experts, shape.hidden, shape.intermediate, r3::moe_backend::format_name(d.weight.format),
                     d.weight.scale_dtype, d.weight.block_n, d.weight.block_k, backend.name,
                     rc == MOE_OK ? "supported" : rc == MOE_UNSUPPORTED_FORMAT ? "weight format not supported" : "geometry not supported");
    return rc == MOE_OK;
}

/* Bytes one expert takes on the device. With a GPU this is the backend contract's size for the container's own description of the
 * expert (moe_backend.h), the same number for every weight format; without one it is the container's stored size. */
int64_t laid_out_bytes(const Container& f, bool device_layout, uint32_t l, uint32_t e) {
    if (!device_layout) return canonical_expert_bytes(f, l, e);
    MoeExpertDesc d{};
    if (!describe_expert(f, l, e, &d)) return -1;
    return r3::moe_backend::expert_bytes(d);
}

bool select_owned(const Container& f, const OwnerPolicy& p, const std::string& arch, bool device_layout,
                  const uint64_t* budgets, uint32_t helper, uint32_t helpers, Owned* out) {
    const uint64_t budget = budgets && helper < helpers ? budgets[helper] : 0;   /* this helper's; the others' matter to the tail placement */
    if (!out || !budget) {
        std::fprintf(stderr, "r3_moe_worker: no helper budget\n");
        return false;
    }
    if (!backend_accepts(f, arch.c_str(), p.formats.c_str(), false)) return false;
    std::vector<const RadFileProfile*> order;
    /* A container converted without calibration has no profile: take experts expert-major (unsorted)
     * so the budget spreads over all layers. */
    std::vector<RadFileProfile> flat;
    /* R3_MOE_WORKER_HEAT: route counts a profiling run wrote (R3_MOE_WORKER_HISTOGRAM). Coldest first, so the
     * engine keeps the experts most tokens route to and a decode step hands off as little as possible. */
    if (const char* hp = p.heat.c_str(); hp && *hp) {
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
    if (p.balance) {
        if (helpers != 2 || flat.empty() || budgets[0] != budgets[1]) {
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_BALANCE needs R3_MOE_WORKER_HEAT, two helpers and one budget for both\n");
            return false;
        }
        if (const char* pv = p.placement.c_str(); pv && *pv && std::strcmp(pv, "roundrobin") != 0) {
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_BALANCE splits the experts of two equal helpers; it does not combine with R3_MOE_WORKER_PLACEMENT=%s\n", pv);
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
        double speed[2] = {p.balance_speeds[0], p.balance_speeds[1]};
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
                const int64_t bytes = laid_out_bytes(f, device_layout, l, e);
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
    /* R3_MOE_WORKER_SHARES="w0,w1,..": the helpers' parts of the ranking (default equal). A slower card takes a smaller part; a part of 0
     * is a helper that owns nothing. Every helper reads the same list and takes the same decision (worker_share.h). */
    uint32_t shares[r3::worker_share::kMaxHelperShares] = {};
    for (uint32_t i = 0; i < helpers; ++i)
        shares[i] = (uint32_t)p.shares.at(i);
    /* R3_MOE_WORKER_PLACEMENT: "roundrobin" (default) deals the ranking out by the shares; "tail" deals it to the head helpers 0..H-2 by their
     * shares, each up to its own budget, and gives the LAST helper what no head helper had room for (RAD23: the cold tail on a small card). */
    bool tail = false;
    if (const char* pv = p.placement.c_str(); pv && *pv) {
        if (std::strcmp(pv, "tail") == 0) tail = true;
        else if (std::strcmp(pv, "roundrobin") != 0) {
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_PLACEMENT is roundrobin or tail, not %s\n", pv);
            return false;
        }
        if (tail && helpers < 2) {
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_PLACEMENT=tail needs at least two helpers\n");
            return false;
        }
    }
    if (!tail && shares[helper] == 0) {
        std::fprintf(stderr, "r3_moe_worker: helper %u has share 0 and owns no experts\n", helper);
        return true;
    }
    if (tail) {
        std::vector<std::pair<uint32_t, uint32_t>> ranked;
        std::vector<int64_t> sizes;
        std::vector<uint8_t> once((size_t)f.layers * f.experts, 0);
        for (const RadFileProfile* p : order) {
            const uint32_t l = (uint32_t)p->layer, e = (uint32_t)p->expert;
            uint8_t& w = once[(size_t)l * f.experts + e];
            if (w) continue;
            w = 1;
            const int64_t bytes = laid_out_bytes(f, device_layout, l, e);
            if (bytes <= 0) {
                std::fprintf(stderr, "r3_moe_worker: expert (%u,%u) cannot be sized for the device\n", l, e);
                return false;
            }
            ranked.emplace_back(l, e);
            sizes.push_back(bytes);
        }
        const std::vector<int32_t> plan = r3::worker_share::plan_tail(sizes, helpers, shares, budgets);
        for (size_t i = 0; i < ranked.size(); ++i)
            if (plan[i] == (int32_t)helper) {
                out->expert.push_back(ranked[i]);
                out->bytes += (uint64_t)sizes[i];
            }
        std::fprintf(stderr, "r3_moe_worker: tail placement: helper %u of %u owns %zu of %zu ranked experts, %.2f GiB of its %.2f GiB\n",
                     helper, helpers, out->expert.size(), ranked.size(), (double)out->bytes / (1u << 30), (double)budget / (1u << 30));
        return true;
    }
    std::vector<uint8_t> seen((size_t)f.layers * f.experts, 0);
    uint64_t position = 0;
    for (const RadFileProfile* p : order) {
        const uint32_t l = (uint32_t)p->layer, e = (uint32_t)p->expert;
        uint8_t& was = seen[(size_t)l * f.experts + e];
        if (was) continue;
        was = 1;
        /* Helper h takes its share of the candidates: disjoint sets, each spread over the same order. */
        if (r3::worker_share::owner_of(position++, shares, helpers) != helper) continue;
        const int64_t bytes = laid_out_bytes(f, device_layout, l, e);
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

bool plan_owners(const Container& f, const std::vector<OwnerPolicy>& policies, const std::vector<std::string>& archs,
                 const std::vector<uint64_t>& budgets, OwnerMap* out, std::string* why) {
    const size_t n = policies.size();
    if (!out || n == 0 || archs.size() != n || budgets.size() != n || n > r3::worker_share::kMaxHelperShares) {
        if (why) *why = "plan_owners: one policy, card and budget per helper, at most " +
                        std::to_string(r3::worker_share::kMaxHelperShares);
        return false;
    }
    *out = OwnerMap{};
    uint64_t b[r3::worker_share::kMaxHelperShares] = {};
    for (size_t i = 0; i < n; ++i) b[i] = budgets[i];
    std::vector<uint8_t> owned((size_t)f.layers * f.experts, 0);
    for (uint32_t h = 0; h < n; ++h) {
        Owned o;
        if (!select_owned(f, policies[h], archs[h], true, b, h, (uint32_t)n, &o)) {
            if (why) *why = "helper " + std::to_string(h) + " (" + archs[h] + "): owned-expert selection refused";
            return false;
        }
        for (auto [l, e] : o.expert) {
            uint8_t& w = owned[(size_t)l * f.experts + e];
            if (w) {
                if (why) *why = "helpers overlap at expert (" + std::to_string(l) + "," + std::to_string(e) + ")";
                return false;
            }
            w = 1;
        }
        out->helpers.push_back(std::move(o));
    }
    for (uint32_t l = 0; l < f.layers; ++l)
        for (uint32_t e = 0; e < f.experts; ++e)
            if (f.weight(l, e, 0)) {
                ++out->routed;
                if (!owned[(size_t)l * f.experts + e]) out->unowned.emplace_back(l, e);
            }
    return true;
}

}  // namespace bc::experts
