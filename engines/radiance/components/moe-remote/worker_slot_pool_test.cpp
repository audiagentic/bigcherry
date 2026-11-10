#include "worker_slot_pool.h"
#include "worker_footprint.h"
#include "worker_generation_tables.h"
#include "moe_remote_shm.h"
#include "experts.h"
#include "rad_encoding.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <new>
#include <string>
#include <vector>

static void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "r3_worker_slot_pool_test: FAILED %s\n", why); std::exit(1); }
}
struct FakeAllocator {
    std::map<void*, uint64_t> live;
    uint64_t charged = 0;
    size_t attempts = 0;
    bool fail = false;
    bool allocate(uint64_t n, void** out) {
        ++attempts;
        if (fail || !n || n > SIZE_MAX) return false;
        *out = new(std::nothrow) char[size_t(n)];
        if (!*out) return false;
        live.emplace(*out, n); charged += n; return true;
    }
    void release(void* p) {
        auto it = live.find(p); check(it != live.end(), "release exactly once");
        charged -= it->second; live.erase(it);
        delete[] static_cast<char*>(p);
    }
};
using r3::residency::Result;
using r3::worker_slots::Pool;
using r3::worker_tables::Model;

static void shape(uint32_t ranks, uint32_t helpers, uint32_t tail) {
    for (uint32_t h = 0; h < helpers; ++h) {
        FakeAllocator alloc;
        const uint32_t selected = h == tail ? 32 : 64;
        const uint64_t budget = selected * 100;
        {
            Pool<FakeAllocator> pool(budget, 100, selected, alloc);
            check(pool.valid() && pool.capacity() == selected &&
                  pool.ledger().headroom() == 100, "budget reserves only weight slots");
            for (uint32_t e = 0; e < selected; ++e) {
                uint32_t id = UINT32_MAX;
                check(pool.bootstrap(e, 100, 1, &id) == Result::OK &&
                      id == e && pool.address(id) != nullptr, "distinct expert weight slots");
            }
            uint32_t id = UINT32_MAX;
            const size_t attempts = alloc.attempts;
            check(pool.bootstrap(99 + selected, 100, 1, &id) == Result::DEFERRED_CAPACITY &&
                  alloc.attempts == attempts, "one incoming slot cannot be used at bootstrap");
            check(pool.bootstrap(0, 100, 1, &id) != Result::OK, "duplicate expert rejected");
            check(pool.ledger().charged() == budget && pool.arena_bytes() == budget &&
                  alloc.charged == budget + 100 && alloc.attempts == 2 &&
                  pool.address(pool.staging_id()) != pool.address(0),
                  "selected-only arena and independent staging allocation");
            check(pool.ledger().reserve(selected + 1, 100, 2, true, &id) == Result::OK &&
                  id == pool.staging_id() && pool.address(id),
                  "one staged move can reserve outside weight budget");
            check(pool.ledger().reserve(selected + 2, 100, 2, true, &id) ==
                  Result::DEFERRED_CAPACITY, "second incoming expert refused");
        }
        check(alloc.live.empty() && !alloc.charged, "all device slots freed");
    }
    check(ranks == 1 || (ranks == 2 && helpers == 1), "rank topology supported");
}
static void four_plane_tables() {
    struct Planes {
        uintptr_t gate_code, gate_scale, down_code, down_scale;
    };
    Model<Planes> table(2, 3);
    std::vector<Planes> old(6), next(6);
    for (size_t i = 0; i < old.size(); ++i) {
        old[i] = Planes{uintptr_t(i*4+1), uintptr_t(i*4+2),
                        uintptr_t(i*4+3), uintptr_t(i*4+4)};
        next[i] = Planes{uintptr_t(i*4+101), uintptr_t(i*4+102),
                         uintptr_t(i*4+103), uintptr_t(i*4+104)};
    }
    check(table.build(0, old) && table.build(1, next), "four-plane generations built");
    for (uint32_t layer = 0; layer < 2; ++layer)
        for (uint32_t expert = 0; expert < 3; ++expert) {
            Planes a{}, b{};
            const size_t index = size_t(layer) * 3 + expert;
            check(table.select(0, layer, expert, &a) &&
                  table.select(1, layer, expert, &b) &&
                  a.gate_code == old[index].gate_code &&
                  a.gate_scale == old[index].gate_scale &&
                  a.down_code == old[index].down_code &&
                  a.down_scale == old[index].down_scale &&
                  b.gate_code == next[index].gate_code &&
                  b.gate_scale == next[index].gate_scale &&
                  b.down_code == next[index].down_code &&
                  b.down_scale == next[index].down_scale,
                  "no four-plane cross-generation alias");
        }
}



static void make_model(bc::experts::Container& f, uint32_t layers) {
    const auto shape = r3::moe_shm::legacy_shape();
    f.layers = layers; f.experts = shape.experts; f.hidden = shape.hidden;
    f.intermediate = shape.intermediate; f.top_k = shape.top_k;
    f.enc_store.resize(1);
    RadEncoding& enc = f.enc_store[0];
    rad_enc_clear(&enc); rad_enc_copy_str(enc.scheme, "w4nl");
    rad_enc_add_plane(&enc, "codes", RAD_U4, 1, 1);
    rad_enc_add_table(&enc, "table", RAD_F16, 1, 16);
    rad_enc_add_plane(&enc, "scale", RAD_F8E4M3, 1, 64);
    check(rad_enc_valid(&enc), "Flash-Next W4 encoding");
    f.encs = f.enc_store.data();
    f.dir_store.resize(size_t(layers) * f.experts * 2);
    f.expert.resize(size_t(layers) * f.experts);
    for (uint32_t l = 0; l < layers; ++l)
        for (uint32_t e = 0; e < f.experts; ++e) {
            for (uint32_t slot = 0; slot < 2; ++slot) {
                RadFileEntry& d = f.dir_store[(size_t(l)*f.experts+e)*2+slot];
                d.enc = 0; d.layer = int32_t(l); d.expert = int32_t(e);
                d.slot = int32_t(slot); d.rank = 2;
                d.shape[0] = slot == 0 ? int64_t(f.intermediate)*2 : f.hidden;
                d.shape[1] = slot == 0 ? f.hidden : f.intermediate;
                f.expert[size_t(l)*f.experts+e][slot] = &d;
            }
            f.profile_store.push_back(RadFileProfile{int32_t(l), int32_t(e),
                float((uint64_t(l)*f.experts+e) % 16001u), 0.0f});
        }
    f.dir = f.dir_store.data();
    f.profile = f.profile_store.data();
    f.hdr.prof_count = f.profile_store.size();
}
struct HelperConfig {
    uint64_t budget = 0;
    std::string arch, placement = "head";
    uint32_t share = 1, count = 1;
};
static void production_layout_parity() {
    const auto s = r3::moe_shm::legacy_shape();
    constexpr uint32_t layers = 49; // Flash-Next routed layer count
    bc::experts::Container f;
    make_model(f, layers);
    const auto pad = [](uint64_t n) { return (n+255u)&~uint64_t(255u); };
    const uint64_t stride = pad(uint64_t(s.intermediate)*s.hidden) +
        pad(uint64_t(2)*s.intermediate*(s.hidden/64)) +
        pad(uint64_t(s.hidden)*s.intermediate/2) +
        pad(uint64_t(s.hidden)*(s.intermediate/64));
    check(stride == 2534400 &&
          uint64_t(bc::experts::laid_out_bytes(f, true, 0, 0)) == stride,
          "real model's padded slot equals main's expert bytes");
    std::ifstream in(R3_LAYOUTS_TOML);
    check(in.good(), "layouts.toml readable");
    std::map<std::string,std::vector<HelperConfig>> layouts;
    std::string name, line;
    HelperConfig group;
    bool open = false;
    auto flush = [&] {
        if (!open) return;
        check(!group.arch.empty(), "valid helper group");
        for (uint32_t i=0; i<group.count; ++i)
            layouts[name].push_back({group.budget, group.arch, group.placement, group.share, 1});
        open=false;
    };
    while (std::getline(in,line)) {
        if (line.rfind("[[layout.",0)==0) {
            flush();
            size_t end=line.find(".helper]]");
            check(end!=std::string::npos,"helper stanza");
            name=line.substr(9,end-9); group=HelperConfig{}; open=true;
        } else if (line.rfind("[layout.",0)==0) {
            flush();
        } else if (open) {
            size_t eq=line.find('=');
            if(eq==std::string::npos) continue;
            auto key=line.substr(0,eq);
            size_t q=line.find('"'), end=q==std::string::npos ? q : line.find('"',q+1);
            if(key=="count ") group.count = line.find("all")!=std::string::npos ?
                2u : uint32_t(std::stoul(line.substr(eq+1)));
            if(key=="budget-mib ") group.budget=std::stoull(line.substr(eq+1))<<20;
            if(key=="share ") group.share=uint32_t(std::stoul(line.substr(eq+1)));
            if(key=="arch " && end!=std::string::npos) group.arch=line.substr(q+1,end-q-1);
            if(key=="placement " && end!=std::string::npos) group.placement=line.substr(q+1,end-q-1);
        }
    }
    flush(); check(!layouts.empty(),"production layouts present");
    for(const auto& [name,helpers]:layouts) {
        check(!helpers.empty() && helpers.size()<=r3::moe_shm::kMaxHelpers,
              "production layout helper count");
        std::vector<uint64_t> budgets;
        std::vector<std::string> archs;
        std::vector<bc::experts::OwnerPolicy> policies;
        std::string shares;
        for(const auto& h: helpers) {
            if(!shares.empty()) shares+=",";
            shares+=std::to_string(h.share);
        }
        for(uint32_t i=0; i<helpers.size(); ++i) {
            const auto& h = helpers[i];
            uint64_t budget = h.budget;
            if (!budget) {
                // Test inventory and the same footprint arithmetic as bc-serve.
                // Neither the production recipe nor this test supplies a budget literal.
                const uint64_t card_mib = h.arch=="gfx1100" ? 24560 :
                                          h.arch=="gfx1030" ? 16368 :
                                          h.arch=="gfx1201" ? 32624 : 0;
                const uint64_t reserve_mib = h.arch=="gfx1201" ? 2048 : 1024;
                check(card_mib > reserve_mib, "derived fixture device/arch");
                namespace fp = r3::worker_footprint;
                const fp::Shape shape{layers,s.experts,s.hidden,s.intermediate,s.top_k,r3::moe_shm::kActGroup};
                const auto classes = fp::small_classes({},8,4096);
                const auto lanes = fp::lanes_for(classes,4096,uint32_t(helpers.size()),i,1,true);
                const uint64_t fixed = fp::fixed_bytes(shape,lanes,fp::stride_bound(stride));
                budget = fp::expert_budget_bytes(card_mib<<20,reserve_mib<<20,fixed) & ~uint64_t((1u<<20)-1);
            }
            check(budget > 0,"derived production helper budget");
            budgets.push_back(budget); archs.push_back(h.arch);
            bc::experts::OwnerPolicy p;
            p.shares=shares; p.formats="w4nl64";
            p.placement=helpers.back().placement=="tail" ? "tail" : "roundrobin";
            policies.push_back(p);
        }
        bc::experts::OwnerMap manifest;
        std::string why;
        check(bc::experts::plan_owners(f,policies,archs,budgets,&manifest,&why),
              "manifest plans production helper owners");
        for(uint32_t h=0;h<helpers.size();++h) {
            bc::experts::Owned worker;
            check(bc::experts::select_owned(f,policies[h],archs[h],true,{},
                budgets.data(),h,uint32_t(helpers.size()),&worker),
                "worker selects production helper owners");
            check(worker.expert==manifest.helpers[h].expert &&
                  worker.bytes==manifest.helpers[h].bytes,
                  "manifest and worker owners, exact IDs/order and bytes");
            auto plan=r3::worker_slots::budget_plan(budgets[h],stride,
                                                   uint64_t(layers)*s.experts);
            check(worker.expert.size()<=plan.expert_slots &&
                  worker.bytes<=budgets[h] && uint64_t(worker.expert.size())*stride<=budgets[h],
                  "production layouts preserve main counts and expert budget");
            // Full-size, allocation-free ledger proof: all planned weights
            // and exactly one separately budgeted incoming expert are admitted.
            r3::residency::Ledger ledger(budgets[h]+stride,0,stride,
                                        uint32_t(worker.expert.size()+1));
            for(uint32_t e=0;e<worker.expert.size();++e) {
                uint32_t id=UINT32_MAX;
                check(ledger.bootstrap(e,stride,1,&id)==Result::OK && id==e,
                      "full-size bootstrap fills every main expert");
            }
            uint32_t id=UINT32_MAX;
            check(ledger.reserve(uint32_t(worker.expert.size()),stride,2,true,&id)==Result::OK &&
                  id==worker.expert.size(), "incoming reserve outside expert budget");
            check(ledger.reserve(uint32_t(worker.expert.size()+1),stride,2,true,&id)==
                  Result::DEFERRED_CAPACITY, "only one incoming slot");
            check(plan.expert_bytes>=worker.expert.size()*stride,
                  "derived production arena equals owned expert allocation");
        }
    }
}
static void empty_bootstrap_can_gain() {
    FakeAllocator alloc;
    {
        Pool<FakeAllocator> pool(300,100,3,alloc);
        check(pool.prepare() && pool.prepare() &&
              alloc.attempts==2 && alloc.charged==400 &&
              pool.arena_bytes()==300 && pool.address(0) &&
              pool.address(2) && pool.address(pool.staging_id()) &&
              pool.ledger().charged()==0,
              "zero-resident helper preallocates full budget and incoming slot");
        uint32_t id=UINT32_MAX;
        check(pool.ledger().reserve(21,100,2,true,&id)==Result::OK &&
              id==0 && pool.address(id) &&
              pool.ledger().start_copy(id)==Result::OK,
              "first arrival uses preallocated address with no bootstrap call");
        for (uint32_t plane=0;plane<4;++plane)
            check(pool.ledger().plane_complete(id,plane)==Result::OK,
                  "all first expert planes completed");
        check(pool.ledger().ready(id)==Result::OK &&
              pool.ledger().activate(id,2)==Result::OK &&
              pool.ledger().charged()==100,
              "first expert activates after its upload");
    }
    check(alloc.live.empty() && !alloc.charged,
          "zero-bootstrap arena and stage released without leak");
}
static void free_memory_guards() {
    using r3::worker_slots::check_device_memory;
    const r3::worker_slots::DeviceMemory m{200,100,50,100,300};
    auto fit=check_device_memory(m,800,600);
    check(fit.fits && fit.total==750 && fit.future==550,
          "already allocated lanes not double-counted against current free memory");
    check(!check_device_memory(m,749,600).fits &&
          check_device_memory(m,749,600).shortfall==1,
          "one-byte shortage vs initial free fails closed");
    check(!check_device_memory(m,800,549).fits &&
          check_device_memory(m,800,549).shortfall==1,
          "one-byte shortage vs current free fails closed");
    check(!check_device_memory({0,UINT64_MAX,1,1,1},UINT64_MAX,UINT64_MAX).fits,
          "overflow fails closed");
}
int main() {
    production_layout_parity();
    empty_bootstrap_can_gain();
    free_memory_guards();
    four_plane_tables();
    shape(2,1,UINT32_MAX);
    shape(1,3,2);
    check(r3::worker_tables::valid_generation(0) &&
          r3::worker_tables::valid_generation(1) &&
          !r3::worker_tables::valid_generation(2),
          "invalid generation rejects");
    FakeAllocator alloc;
    {
        Pool<FakeAllocator> pool(200,100,3,alloc);
        uint32_t id=UINT32_MAX;
        check(!pool.valid() && pool.bootstrap(0,100,1,&id)==Result::INVALID &&
              !alloc.attempts, "overbudget first bootstrap rejects before allocation");
    }
    {
        Pool<FakeAllocator> pool(200,100,2,alloc);
        uint32_t id=UINT32_MAX;
        check(pool.bootstrap(3,101,1,&id)==Result::INVALID && !alloc.attempts,
              "oversized plane layout rejected");
        alloc.fail=true;
        check(pool.bootstrap(0,100,1,&id)!=Result::OK && !pool.ledger().charged(),
              "allocation failure no phantom ledger charge");
    }
    check(alloc.live.empty(),"failure no leak");
    std::puts("r3_worker_slot_pool_test: ok");
}
